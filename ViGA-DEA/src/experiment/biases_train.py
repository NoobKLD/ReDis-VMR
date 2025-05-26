import os
import yaml
import torch
import random
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm

from src.dataset.dataset import prepare_data
from src.utils.utils import load_config, n_params, get_now
from src.utils.vl_utils import GloVe
from src.experiment.eval import Evaluator
from src.model.model import Model
from src.dataset.generate_bias import BiasDataGenerator

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def train(config):
    # save log
    exp_folder_path = os.path.join(config["exp_dir"], get_now())
    Path(exp_folder_path).mkdir(parents=True, exist_ok=True)

    # add a6000 or rtx8000 done
    with open(os.path.join(exp_folder_path, config["config_file"]), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # prepare data
    dataset_name = config["dataset_name"]
    data = prepare_data(config, dataset_name)
    train_dl = data["train_dl"]
    valid_dl = data["valid_dl"]
    test_dl = data["test_dl"]

    # bias相关参数和对象
    bias_generator = BiasDataGenerator(dataset_name=config["dataset_name"])
    bias_train_datasets = []

    vocab = data["vocab"]
    glove = GloVe(glove_path=config["model"]["glove_path"])

    model = Model(config, vocab, glove, without_video_feats=config['use_biases'])
    print("Model has {} parameters.\n".format(n_params(model)))
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.gpu_mode()
    else:
        model.cpu_mode()

    test_evaluator = Evaluator()
    log_file_path = os.path.join(exp_folder_path, "train.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("{}\n".format(config[dataset_name]["feature_dir"]))
        log_file.write("Model has {} parameters.\n".format(n_params(model)))
        log_file.flush()

        if dataset_name == 'charadessta':
            bias_generate_per_epoch = 10
        elif dataset_name == 'activitynetcaptions':
            bias_generate_per_epoch = 15
        else:
            raise ValueError("Dataset name not found.")
        # bias_generate_per_epoch = 1  # test
        # bias_generate_per_epoch = 10  # charades
        # bias_generate_per_epoch = 15  # activitynet
        # bias_generate_per_epoch = 20  # tacos
        use_replay = True
        # use_replay = False
        use_dynamic_model = True
        # use_dynamic_model = False

        for epoch in range(1, config[dataset_name]["epoch"] + 1):

            if use_replay:
                xx_b = bias_train_datasets
            else:
                xx_b = bias_train_datasets[-1:]

            for bias_train_dataset in xx_b:
                for batch in tqdm(bias_train_dataset):
                    model.train_mode()
                    loss = model.forward_train_val(
                        {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    )
                    # print(loss.item())
                    model.optimizer_step(loss)

            if use_replay or not bias_train_datasets:
                for i, batch in tqdm(
                    enumerate(train_dl), total=len(train_dl),
                    desc="Training epoch {} with lr {}".format(epoch, model.optimizer.param_groups[0]["lr"])
                ):
                    # 先训练原始数据
                    model.train_mode()
                    loss = model.forward_train_val(batch)
                    # print(loss.item())
                    model.optimizer_step(loss)

                # loss = model.forward_train_val(
                #     {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # )
                # # print(loss.item())
                # model.optimizer_step(loss)

                # if use_gpu:
                #     model.batch_to(batch, model.cpu_device)
                #     torch.cuda.empty_cache()

            # 用于记录每个视频样本的结果 （名称，预测值，实际值）
            pred_gt_path = os.path.join(exp_folder_path, "pred_gt_{}.txt".format(epoch))

            with torch.no_grad():
                new_bias_generator = bias_generator if epoch > 1 and (epoch - 1) % bias_generate_per_epoch == 0 else None
                test_loss, preds, gts, names = test_evaluator.eval_dataloader(model, test_dl, epoch, new_bias_generator=new_bias_generator)
                model.scheduler_step(test_loss)

                # 把预测结果和gt写到文件中
                with open(pred_gt_path, 'w') as pred_file:
                    for pred, gt, name in zip(preds, gts, names):
                        pred = pred.squeeze()
                        for idx, video_id in enumerate(name):
                            pred_file.write("{} {} {}\n".format(video_id, pred[idx].tolist(), gt[idx].tolist()))

                # 生成新的bias分布
                if epoch % bias_generate_per_epoch == 0:
                    x_coords = [pred.squeeze()[idx].tolist()[0] for pred, name in zip(preds, names) for idx, _ in enumerate(name)]
                    y_coords = [pred.squeeze()[idx].tolist()[1] for pred, name in zip(preds, names) for idx, _ in enumerate(name)]

                    bias_generator.generate_bias_ditributon(
                        coords=np.vstack([x_coords, y_coords]),
                        sample_size=config[dataset_name]['video_feature_len'],
                        epoch=epoch,
                    )

                    bias_train_datasets.append([])
                    for i, batch in tqdm(
                            enumerate(train_dl), total=len(train_dl),
                            desc="generating training dataset"
                    ):
                        # 生成修改分布的batch
                        batch = bias_generator.sample_point(batch)
                        bias_train_datasets[-1].append(
                            {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        )

                    # 新增持续学习模块
                    if use_dynamic_model:
                        model.continual_model()


            log_file.write("\n==== epoch {} ====\n".format(epoch))
            log_file.write("        ## test ##\n")
            log_file.write(test_evaluator.report_current() + "\n")
            log_file.write(test_evaluator.report_best() + "\n")
            log_file.flush()

            # save best
            if epoch == test_evaluator.best_epoch:
                model.save_checkpoint(exp_folder_path, "best")
            print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--task", help="Dataset name in {activitynetcaptions, charadessta, tacos}.", required=True)
    parser.add_argument("--biases", type=bool, default=False)
    parser.add_argument("--config-file", type=str, default="config.yaml")
    args = parser.parse_args()

    if args.config_file == 'config_3090.yaml':
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    config = load_config("src/{}".format(args.config_file))
    config["dataset_name"] = args.task
    config["use_biases"] = args.biases
    config["config_file"] = args.config_file
    train(config)
