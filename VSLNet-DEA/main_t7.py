import os
import time
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from model.VSLNet_t7 import VSLNet, build_optimizer_and_scheduler
from util.data_util import load_video_features, save_json, load_json
from util.data_gen import gen_or_load_dataset
from util.data_loader_t7 import get_train_loader, get_test_loader
from util.runner_utils_t7 import set_th_config, convert_length_to_mask, eval_test, filter_checkpoints, \
    get_last_checkpoint

from generate_bias import BiasDataGenerator

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='datasets_t7', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='charades', help='target task')
parser.add_argument('--fv', type=str, default='new', help='[new | org] for visual features')
parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed')
# model parameters
parser.add_argument("--word_size", type=int, default=None, help="number of words")
parser.add_argument("--char_size", type=int, default=None, help="number of characters")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension")
parser.add_argument("--char_dim", type=int, default=50, help="character dimension, set to 100 for activitynet")
parser.add_argument("--dim", type=int, default=128, help="hidden size")
parser.add_argument("--highlight_lambda", type=float, default=5.0, help="lambda for highlight region")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
parser.add_argument('--predictor', type=str, default='rnn', help='[rnn | transformer]')
# training/evaluation parameters
parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
parser.add_argument("--seed", type=int, default=12345, help="random seed")
parser.add_argument("--mode", type=str, default="train", help="[train | test]")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--extend", type=float, default=0.1, help="highlight region extension")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument('--model_dir', type=str, default='ckpt_t7', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='vslnet', help='model name')
parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')
parser.add_argument('--settings', type=str, default='ReDis', help='set to "ReDis" or "traditional"')

configs = parser.parse_args()

if configs.settings == 'ReDis':
    use_original_setting = False
    original_setting_epoch = 1
    bias_per_epoch = 30 if configs.task == 'charades' else 50
    use_dynamic = True
    use_replay = True
    use_fair = True
elif configs.settings == 'traditional':
    use_original_setting = True
    original_setting_epoch = 30 if not configs.task.endswith('_cd') else 1
    bias_per_epoch = 99999
    use_dynamic = False
    use_replay = False
    use_fair = False
else:
    raise ValueError("Invalid setting name")

# add bias generator
bias_generator = BiasDataGenerator(dataset_name=configs.task)
# bias_per_epoch = 1  # test
# bias_per_epoch = 5  # test2
# bias_per_epoch = 30  # charades and tacos
# bias_per_epoch = 50  # activitynet
# bias_per_epoch = 99999  # original setting for all datasets

# use_original_setting = False
# use_original_setting = True
# original_setting_epoch = 1  # test and for charades_cd, activitynet_cd
# original_setting_epoch = 10 #
# original_setting_epoch = 30  #   charades,  activitynet
# original_setting_epoch = 50  #

# if use_original_setting:
#     bias_per_epoch = 99999

# use_dynamic = False
# use_dynamic = True
# use_replay = False
# use_replay = True
# use_fair = False
# use_fair = True


assert not (use_original_setting and use_replay)
assert not (use_original_setting and use_dynamic)

# set tensorflow configs
set_th_config(configs.seed)

# prepare or load dataset
dataset = gen_or_load_dataset(configs)
configs.char_size = dataset['n_chars']
configs.word_size = dataset['n_words']

# get train and test loader
visual_features = load_video_features(os.path.join('data', 'features', configs.task, configs.fv), configs.max_pos_len)
train_loader = get_train_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs)
val_loader = None if dataset['val_set'] is None else get_test_loader(dataset['val_set'], visual_features, configs)
test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs)
configs.num_train_steps = len(train_loader) * configs.epochs
num_train_batches = len(train_loader)
num_val_batches = 0 if val_loader is None else len(val_loader)
num_test_batches = len(test_loader)

# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# create model dir
home_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name, configs.task, configs.fv,
                                                     str(configs.max_pos_len), configs.predictor]))
if configs.suffix is not None:
    home_dir = home_dir + '_' + configs.suffix
model_dir = os.path.join(home_dir, "model")

# train and test
if configs.mode.lower() == 'train':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # eval_period = num_train_batches // 2
    eval_period = num_train_batches
    save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    # build model
    model = VSLNet(configs=configs, word_vectors=dataset['word_vector']).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs)
    # start training
    best_r1i7 = -1.0
    time_file = time.time()
    log_path = os.path.join(model_dir, "eval_results_{}.txt".format(time_file))
    print("log saved in {}".format(log_path))
    score_writer = open(log_path, mode="w", encoding="utf-8")
    print('start training...', flush=True)
    global_step = 0
    for epoch in range(configs.epochs):
        model.train()

        if use_dynamic and epoch == bias_per_epoch + 1:
            # print('aaaa')
            model.dynamic_model(device=device)

        for data in tqdm(train_loader, total=num_train_batches, desc='Epoch %3d / %3d' % (epoch + 1, configs.epochs)):
            global_step += 1
            _, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels = data

            if epoch + 1 > bias_per_epoch or (use_original_setting and epoch > original_setting_epoch):

                # if use_replay:
                if use_replay or (use_original_setting and epoch > original_setting_epoch):
                    # prepare features
                    vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
                    word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                    s_labels, e_labels, h_labels = s_labels.to(device), e_labels.to(device), h_labels.to(device)
                    # generate mask
                    query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
                    video_mask = convert_length_to_mask(vfeat_lens).to(device)
                    # compute logits
                    h_score, start_logits, end_logits = model(word_ids, char_ids, vfeats, video_mask, query_mask)
                    # compute loss
                    highlight_loss = model.compute_highlight_loss(h_score, h_labels, video_mask)
                    loc_loss = model.compute_loss(start_logits, end_logits, s_labels, e_labels)

                    fair_loss = model.compute_fair_loss(pre=start_logits,
                                                        tar=torch.ones(start_logits.shape).to(device) /
                                                            start_logits.shape[1]) + \
                                model.compute_fair_loss(pre=end_logits,
                                                        tar=torch.ones(end_logits.shape).to(device) / end_logits.shape[
                                                            1])
                    if use_fair:
                        total_loss = loc_loss + configs.highlight_lambda * highlight_loss + 0.001 * fair_loss / 2.
                    else:
                        total_loss = loc_loss + configs.highlight_lambda * highlight_loss
                    # compute and apply gradients
                    optimizer.zero_grad()
                    total_loss.backward()
                    if not use_original_setting:
                        nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)  # clip gradient
                        optimizer.step()
                        scheduler.step()

                    _, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels = data
                # print(s_labels)
                # print(s_labels.shape)
                # print(e_labels)
                # print(e_labels.shape)
                # print(h_labels)
                # print(h_labels.shape)
                sample_size = vfeats.shape[1]
                vfeat_lens = vfeat_lens.clone()
                vfeat_lens[:] = sample_size
                # for i in range(len(records)):
                #     records[i]['v_len'] = sample_size
                _ = [i.copy() for i in _]
                _, vfeats, s_labels, e_labels, h_labels = bias_generator.sample_point(_, vfeats, configs.max_pos_len, s_labels, e_labels, h_labels)
                # print(s_labels)
                # print(s_labels.shape)
                # print(e_labels)
                # print(e_labels.shape)
                # print(h_labels)
                # print(h_labels.shape)
                # print()
                # exit()

            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            s_labels, e_labels, h_labels = s_labels.to(device), e_labels.to(device), h_labels.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            # compute logits
            h_score, start_logits, end_logits = model(word_ids, char_ids, vfeats, video_mask, query_mask)
            # compute loss
            highlight_loss = model.compute_highlight_loss(h_score, h_labels, video_mask)
            loc_loss = model.compute_loss(start_logits, end_logits, s_labels, e_labels)

            fair_loss = model.compute_fair_loss(pre=start_logits, tar=torch.ones(start_logits.shape).to(device) / start_logits.shape[1]) + \
                        model.compute_fair_loss(pre=end_logits, tar=torch.ones(end_logits.shape).to(device) / end_logits.shape[1])

            if use_fair:
                total_loss = loc_loss + configs.highlight_lambda * highlight_loss + 0.001 * fair_loss / 2.
            else:
                total_loss = loc_loss + configs.highlight_lambda * highlight_loss

            if use_original_setting and (epoch > original_setting_epoch or configs.task.endswith('_cd') or configs.task == 'activitynet'):
                # total_loss *= 0.05   # for charades_cd, activitynet_cd
                # total_loss *= 0.01   # for charades
                total_loss *= 0.1   # for activitynet
            else:
                # compute and apply gradients
                optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)  # clip gradient
            optimizer.step()
            scheduler.step()
            # evaluate
    # if global_step % eval_period == 0 or global_step % num_train_batches == 0:
        model.eval()
        out_put_dir = os.path.join(model_dir, "{}".format(time_file))
        if not os.path.exists(out_put_dir):
            os.mkdir(out_put_dir)
        with open(os.path.join(out_put_dir, "{}_epoch{}.txt".format(time_file, epoch)), 'w') as f:
            r1i3, r1i5, r1i7, mi, score_str = eval_test(model=model, data_loader=test_loader, device=device,
                                                        mode='test', epoch=epoch + 1, global_step=global_step,
                                                        bias_generator=bias_generator, sample_size=configs.max_pos_len,
                                                        bias_per_epoch=99999, file_write=f,
                                                        train=False
                                                        # train=True
                                                        )
        print('\nEpoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % (
            epoch + 1, global_step, r1i3, r1i5, r1i7, mi), flush=True)
        score_writer.write(score_str)
        score_writer.flush()

        if not use_original_setting and (epoch + 1) % bias_per_epoch == 0:
            eval_test(model=model, data_loader=train_loader, device=device,
                      mode='test', epoch=epoch, global_step=global_step,
                      bias_generator=bias_generator, sample_size=configs.max_pos_len,
                      bias_per_epoch=bias_per_epoch,
                      train=False)

        if use_original_setting and epoch == original_setting_epoch:
        # if epoch >= original_setting_epoch:
            eval_test(model=model, data_loader=train_loader, device=device,
                      mode='train', epoch=epoch + 1, global_step=global_step,
                      bias_generator=bias_generator, sample_size=configs.max_pos_len,
                      bias_per_epoch=1, train=True)

        if r1i7 >= best_r1i7:
            best_r1i7 = r1i7
            torch.save(model.state_dict(), os.path.join(model_dir, '{}_{}.t7'.format(configs.model_name,
                                                                                     global_step)))
            # only keep the top-3 model checkpoints
            filter_checkpoints(model_dir, suffix='t7', max_to_keep=3)
        model.train()
    score_writer.close()

elif configs.mode.lower() == 'test':
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    # load previous configs
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # build model
    model = VSLNet(configs=configs, word_vectors=dataset['word_vector']).to(device)
    # get last checkpoint file
    filename = get_last_checkpoint(model_dir, suffix='t7')
    model.load_state_dict(torch.load(filename))
    model.eval()
    r1i3, r1i5, r1i7, mi, _ = eval_test(model=model, data_loader=test_loader, device=device, mode='test')
    print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m", flush=True)
