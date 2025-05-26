import os
import glob
import random
import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn
from tqdm import tqdm
from util.data_util import index_to_time


def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def filter_checkpoints(model_dir, suffix='t7', max_to_keep=5):
    model_paths = glob.glob(os.path.join(model_dir, '*.{}'.format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(os.path.basename(model_path).split('_')[1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def get_last_checkpoint(model_dir, suffix='t7'):
    model_filenames = glob.glob(os.path.join(model_dir, '*.{}'.format(suffix)))
    model_file_dict = dict()
    suffix_len = len(suffix) + 1
    for model_filename in model_filenames:
        step = int(os.path.basename(model_filename).split('_')[1][0:-suffix_len])
        model_file_dict[step] = model_filename
    sorted_tuples = sorted(model_file_dict.items())
    last_checkpoint = sorted_tuples[-1]
    return last_checkpoint[1]


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(lengths.size()[0], max_len) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)


bias_data_loaders = [[]]


def eval_test(
        model, data_loader, device,
        mode='test', epoch=None, global_step=None,
        bias_generator=None, sample_size=128, bias_per_epoch=10, file_write=None, train=False):
    ious = []
    with torch.no_grad():
        start_idx_bias = []
        end_idx_bias = []
        # for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
        #         enumerate(data_loader), total=len(data_loader), desc='evaluate {}'.format(mode)):
        # 验证获得的dataloder 可能是训练集可能是测试集
        for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc='evaluate {}'.format(mode)):
            records, vfeats, vfeat_lens, word_ids, char_ids = batch[:5]
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            # compute predicted results
            _, start_logits, end_logits = model(word_ids, char_ids, vfeats, video_mask, query_mask)
            start_indices, end_indices = model.extract_index(start_logits, end_logits)
            start_indices = start_indices.cpu().numpy()
            end_indices = end_indices.cpu().numpy()
            if (epoch + 1) % bias_per_epoch == 0 or train:
                start_idx_bias.append(start_indices)
                end_idx_bias.append(end_indices)
            for record, start_index, end_index in zip(records, start_indices, end_indices):
                start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
                iou = calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]])
                ious.append(iou)

                if file_write:
                    file_write.write("({}, {}), {}, ({}, {})\n".format(start_time, end_time, record["duration"], start_time, end_time))

        # r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
        # r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
        # r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
        # mi = np.mean(ious) * 100.0
        # # write the scores
        # score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
        # score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
        # score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
        # score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
        # score_str += "mean IoU: {:.2f}\n".format(mi)
        # ious = []
        # 如果redis生成测试集   或者   使用原设定  就获得分布
        if (epoch + 1) % bias_per_epoch == 0 or train:
            start_idx_bias = np.concatenate(start_idx_bias, axis=0)
            end_idx_bias = np.concatenate(end_idx_bias, axis=0)
            bias_generator.generate_bias_ditributon(
                coords=np.vstack([start_idx_bias / sample_size, end_idx_bias / sample_size]),
                sample_size=sample_size,
                epoch=epoch,
                reverse=True
                # reverse=False
            )
            # bias_data_loaders.append([])
            # for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
            #         enumerate(data_loader), total=len(data_loader), desc='generate bias'):
            # 不使用原设定  就生成新验证
            if (epoch + 1) % bias_per_epoch == 0 and not train:
                bias_data_loaders[-1] = []
                for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc='generate bias'):
                    records, vfeats, vfeat_lens, word_ids, char_ids = batch[:5]
                    # print(vfeats.shape)
                    # print(vfeat_lens.shape)
                    # print(records)
                    sample_size = vfeats.shape[1]
                    vfeat_lens_ = vfeat_lens.clone()
                    vfeat_lens_[:] = sample_size
                    # for i in range(len(records)):
                    #     records[i]['v_len'] = sample_size
                    records_ = [i.copy() for i in records]
                    records_, vfeats_ = bias_generator.sample_point(records_, vfeats.clone(), sample_size)
                    # print(vfeats.shape)
                    # print(vfeat_lens.shape)
                    # exit(0)
                    bias_data_loaders[-1].append((records_, vfeats_, vfeat_lens_, word_ids, char_ids))
                    # exit()
        # elif bias_per_epoch * 2 == epoch:
        #     # 只是为了看一下预测分布和test_dis的kld
        #     start_idx_bias = np.concatenate(start_idx_bias, axis=0)
        #     end_idx_bias = np.concatenate(end_idx_bias, axis=0)
        #     bias_generator.generate_bias_ditributon(
        #         coords=np.vstack([start_idx_bias / sample_size, end_idx_bias / sample_size]),
        #         sample_size=sample_size,
        #         epoch=epoch
        #     )
        if bias_data_loaders[-1] and epoch > bias_per_epoch:
            for bias_loader in bias_data_loaders:
                for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
                        enumerate(bias_loader), total=len(bias_loader), desc='evaluate bias loader'):
                    # prepare features
                    vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
                    word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                    # generate mask
                    query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
                    video_mask = convert_length_to_mask(vfeat_lens).to(device)
                    # compute predicted results
                    _, start_logits, end_logits = model(word_ids, char_ids, vfeats, video_mask, query_mask)
                    start_indices, end_indices = model.extract_index(start_logits, end_logits)
                    start_indices = start_indices.cpu().numpy()
                    end_indices = end_indices.cpu().numpy()
                    for record, start_index, end_index in zip(records, start_indices, end_indices):
                        start_time, end_time = index_to_time(start_index, end_index, record["v_len"],
                                                             record["duration"])
                        iou = calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]])
                        # print([start_time, end_time], [record["s_time"], record["e_time"]], iou)
                        ious.append(iou)

                        if file_write:
                            file_write.write(
                                "({}, {}), {}, ({}, {})\n".format(start_time, end_time, record["duration"], start_time,
                                                                  end_time))
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    # write the scores
    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}\n".format(mi)
    if not train:
        print(score_str)
    return r1i3, r1i5, r1i7, mi, score_str
