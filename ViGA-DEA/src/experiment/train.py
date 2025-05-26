import os
import yaml
import torch
import torch.nn as nn
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
    with open(os.path.join(exp_folder_path, config["config_file"]), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # bias相关参数和对象
    bias_generator = BiasDataGenerator(dataset_name=config["dataset_name"])
    bias_train_datasets = [[]]
    # use_redis = False
    use_redis = True
    if config["dataset_name"] == 'Charades_CD':
        original_setting_epoch = 7
    else:
        original_setting_epoch = 1

    # original_setting_epoch = 1  # for test, activitynet, activitynet-cd, charades
    # original_setting_epoch = 7  # for charades-cd
    # original_setting_epoch = 9

    # prepare data
    dataset_name = config["dataset_name"]
    data = prepare_data(config, dataset_name)
    train_dl = data["train_dl"]
    valid_dl = data["valid_dl"]
    test_dl = data["test_dl"]

    vocab = data["vocab"]
    glove = GloVe(glove_path=config["model"]["glove_path"])

    model = Model(config, vocab, glove)
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
        for epoch in range(1, config[dataset_name]["epoch"] + 1):

            # if use_redis:
            #     for bias_train_dataset in bias_train_datasets:
            #         for batch in tqdm(bias_train_dataset, desc="train redis dataset"):
            #             model.train_mode()
            #             loss = model.forward_train_val(
            #                 {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            #             )
            #             # print(loss.item())
            #             model.optimizer_step(loss)

            for i, batch in tqdm(
                enumerate(train_dl), total=len(train_dl),
                desc="Training epoch {} with lr {}".format(epoch, model.optimizer.param_groups[0]["lr"])
            ):
                model.train_mode()

                loss = model.forward_train_val(batch)

                # if use_redis and dataset_name == 'charadessta':
                #     loss *= 0.5

                model.optimizer.zero_grad()
                loss.backward()

                if use_redis and epoch > original_setting_epoch:
                    # 生成修改分布的batch
                    batch = bias_generator.sample_point(batch)
                    if dataset_name == 'charadessta':
                        loss = 0.05 * model.forward_train_val(batch)
                    elif dataset_name == 'Charades_CD':
                        loss = 0.5 * model.forward_train_val(batch)
                    else:
                        loss = (0.5 ** epoch) * model.forward_train_val(batch)

                    # loss = 0.5 * model.forward_train_val(batch)   # for charades-cd
                    # loss = (0.5 ** epoch) * model.forward_train_val(batch)   # for activitynet
                    # loss = 0.05 * model.forward_train_val(batch)   # for charades,
                    # loss = (0.05 ** epoch) * model.forward_train_val(batch)
                    loss.backward()

                # print(loss.item())
                # model.optimizer_step(loss)

                nn.utils.clip_grad_norm_(model.parameters(), model.config["train"]["clip_norm"])
                model.optimizer.step()

                # if use_gpu:
                #     model.batch_to(batch, model.cpu_device)
                #     torch.cuda.empty_cache()

            # if use_redis and epoch >= original_setting_epoch:
            if use_redis and epoch == original_setting_epoch:    # for charades-cd
                with torch.no_grad():
                    _, preds, gts, names = test_evaluator.eval_dataloader(model, train_dl, epoch, train=True)

                x_coords = [pred.squeeze()[idx].tolist()[0] for pred, name in zip(preds, names) for idx, _ in
                            enumerate(name)]
                y_coords = [pred.squeeze()[idx].tolist()[1] for pred, name in zip(preds, names) for idx, _ in
                            enumerate(name)]

                bias_generator.generate_bias_ditributon(
                    coords=np.vstack([x_coords, y_coords]),
                    sample_size=config[dataset_name]['video_feature_len'],
                    epoch=epoch,
                    reverse=True,    # for charades-cd
                    # reverse=False,
                )

                # del bias_train_datasets[-1]
                # bias_train_datasets.append([])
                # for i, batch in tqdm(
                #         enumerate(train_dl), total=len(train_dl),
                #         desc="generating training dataset"
                # ):
                #     # 生成修改分布的batch
                #     batch = bias_generator.sample_point(batch)
                #     bias_train_datasets[-1].append(
                #         {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                #     )

            with torch.no_grad():
                test_loss, _, _, _ = test_evaluator.eval_dataloader(model, test_dl, epoch)
                model.scheduler_step(test_loss)

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
