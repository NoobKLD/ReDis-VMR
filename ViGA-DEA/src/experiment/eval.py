import os
from collections import OrderedDict

import torch
import argparse

from tqdm import tqdm

from src.dataset.dataset import prepare_data
from src.utils.utils import load_config
from src.utils.vl_utils import GloVe
from src.model.model import Model


class Evaluator(object):
    def __init__(self):
        # self.recall_at = [1, 5]
        self.recall_at = [1]
        self.iou_threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
        # self.iou_threshold = [0.3, 0.5, 0.7]

        self.res = None
        self.epoch = None
        self.score = None

        self.best_res = None
        self.best_epoch = None
        self.best_score = None

        # 记录不同的bias数据
        self.bias_loaders = []

    def _res_to_string(self, res):
        return "\n".join([" | ".join([
            "R@{} IoU {}: {:.2f}".format(recall_at, thres, res[recall_at][thres] * 100) for thres in self.iou_threshold
        ] + ["mIoU: {:.2f}".format(res[recall_at]["miou"] * 100)]) for recall_at in self.recall_at])

    @staticmethod
    def _res_to_score(res):
        """ Method to judge best epoch, currently just sum all metrics up.
        """
        return sum([sum([v for _, v in d.items()]) for _, d in res.items()])

    def _update(self, preds, gts, epoch=0):
        """
        Args:
            preds: (N, topk, 2)
            gts: (N, 2)
        """
        N, topk, _ = preds.shape
        gts = gts.unsqueeze(1)
        intersection = torch.clamp(torch.min(preds[:, :, 1], gts[:, :, 1]) - torch.max(preds[:, :, 0], gts[:, :, 0]), min=0.0)
        union = torch.clamp(torch.max(preds[:, :, 1], gts[:, :, 1]) - torch.min(preds[:, :, 0], gts[:, :, 0]), min=0.0)
        iou = intersection / union
        iou[torch.logical_or(intersection == 0.0, union == 0.0)] = 0.0  # (B, topk)
        miou = torch.mean(iou, dim=0)

        res = OrderedDict()
        for recall_at in self.recall_at:
            d = OrderedDict()
            temp = iou[:, :recall_at]
            for thres in self.iou_threshold:
                count = torch.sum(torch.any(temp >= thres, dim=1).to(torch.long)).item()
                d[thres] = round(count / N, 4)
            d["miou"] = miou[recall_at - 1].item()
            res[recall_at] = d

        # update result and best result
        self.res = res
        self.epoch = epoch
        self.score = self._res_to_score(res)
        if self.best_res is None or self._res_to_score(res) > self._res_to_score(self.best_res):
            self.best_res = res
            self.best_epoch = epoch
            self.best_score = self._res_to_score(res)

    def eval_dataloader(self, model, dataloader, epoch=0, new_bias_generator=None, train=False):
        model.eval_mode()
        preds = []
        gts = []
        names = []
        loss = 0

        # 生成新的bias测试集
        if new_bias_generator:
            self.bias_loaders.append([])
            for batch in tqdm(dataloader, total=len(dataloader), desc="Generating bias {} epoch {}".format(len(self.bias_loaders) - 1, epoch)):
                self.bias_loaders[-1].append(new_bias_generator.sample_point(batch))

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating epoch {}".format(epoch)):
                # if train and batch_idx / len(dataloader) > 0.1:
                #     break
                pred = model.forward_eval(batch)
                loss += model.forward_train_val(batch).item()
                preds.append(pred)
                gt = torch.stack([batch["start_frac"], batch["end_frac"]], dim=1)
                gts.append(gt)
                names.append(batch['video_id'])

            for idx, bias_loader in enumerate(self.bias_loaders):
                for batch in tqdm(bias_loader, total=len(bias_loader), desc="Evaluating bias {} epoch {}".format(idx, epoch)):
                    pred = model.forward_eval(batch)
                    loss += model.forward_train_val(batch).item()
                    preds.append(pred)
                    gt = torch.stack([batch["start_frac"], batch["end_frac"]], dim=1)
                    gts.append(gt)
                    names.append(batch['video_id'])

        preds_list = preds.copy()
        gts_list = gts.copy()

        loss /= len(dataloader)
        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)
        if not train:
            self._update(preds, gts, epoch)
            print(self.report_current())
            print(self.report_best())
        return loss, preds_list, gts_list, names

    def report_current(self):
        return "This epoch {}\n{}".format(self.epoch, self._res_to_string(self.res))

    def report_best(self):
        return "Best epoch {}\n{}".format(self.best_epoch, self._res_to_string(self.best_res))


def evaluate(exp_folder_path):
    config = load_config(os.path.join(exp_folder_path, "config.yaml"))
    data = prepare_data(config, config["dataset_name"])
    test_dl = data["test_dl"]

    vocab = data["vocab"]
    glove = GloVe(glove_path=config["model"]["glove_path"])

    model = Model(config, vocab, glove)
    model.load_checkpoint(exp_folder_path, "best")
    model.eval_mode()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.gpu_mode()
    else:
        model.cpu_mode()

    evaluator = Evaluator()
    evaluator.eval_dataloader(model, test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval trained model.")
    parser.add_argument("--exp", help="Experiment folder to evaluate.", required=True)
    args = parser.parse_args()
    evaluate(args.exp)
