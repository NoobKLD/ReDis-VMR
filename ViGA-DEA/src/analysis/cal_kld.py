import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from torch import nn
from tqdm import tqdm

from src.dataset.dataset import prepare_data
from src.utils.utils import load_config


# from generate_bias import BiasDataGenerator



def cal_candidate_distribution(coords, sample_size):
    # 生成概率密度函数的网格
    x_vals, y_vals = np.meshgrid(
        # np.linspace(a[0].min(), a[0].max(), 100),
        np.linspace(0., 1., sample_size),
        # np.linspace(a[1].min(), a[1].max(), 100)
        np.linspace(0., 1., sample_size)

    )
    iou_mask = np.zeros((sample_size, sample_size)).astype(bool)
    for c in range(coords.shape[1]):
        # iou_mask[int(coords[0, c] * sample_size), int(coords[1, c] * sample_size)] = 1
        iou_mask[int(coords[1, c] * sample_size), int(coords[0, c] * sample_size)] = 1

    positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

    # print(positions.shape)
    # 根据IoU生成对应候选的权重weight并且用于gaussian_kde

    a = coords
    min_s = np.minimum(a[0][:, np.newaxis], positions[0])
    max_s = np.maximum(a[0][:, np.newaxis], positions[0])
    min_e = np.minimum(a[1][:, np.newaxis], positions[1])
    max_e = np.maximum(a[1][:, np.newaxis], positions[1])

    i_se = np.maximum(0, min_e - max_s)
    # print(i_se.sum())
    u_se = np.maximum(1e-9, max_e - min_s)
    # print(u_se.sum())

    iou_se = i_se / u_se
    # print(iou_se.shape)
    # print(iou_se)

    # FIXME iou_mask
    # print(iou_se.shape)
    old_iou_se_shape = iou_se.shape
    # print(iou_se.shape)
    iou_se[:, ~iou_mask.flatten()] = 0.
    # print(iou_se.sum())
    iou_se = iou_se.reshape(old_iou_se_shape)

    # 使用gaussian_kde生成二维概率密度函数
    # kde = gaussian_kde(a, bw_method=50)
    kde = gaussian_kde(positions, bw_method=None, weights=iou_se.sum(0))

    # density = kde(positions).reshape(100, 100)
    density = kde(positions).reshape(sample_size, sample_size)

    # 可视化归一化后的结果
    cmap = 'viridis'
    plt.contourf(x_vals, y_vals, density, cmap=cmap)
    plt.title('Normalized candidate Density Function')
    plt.xlabel('start time (X-axis)')
    plt.ylabel('end time (Y-axis)')
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.colorbar()
    plt.show()
    return density


def cal_distribution(coords, sample_size):
    # 生成概率密度函数的网格
    x_vals, y_vals = np.meshgrid(
        np.linspace(0., 1., sample_size),
        np.linspace(0., 1., sample_size)
    )
    positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

    # print(positions.shape)
    # 根据IoU生成对应候选的权重weight并且用于gaussian_kde

    a = coords


    # 使用gaussian_kde生成二维概率密度函数
    # kde = gaussian_kde(a, bw_method=50)
    kde = gaussian_kde(a)
    # kde = gaussian_kde(positions, bw_method=None, weights=iou_se.sum(0))

    # density = kde(positions).reshape(100, 100)
    density = kde(positions).reshape(sample_size, sample_size)

    # 可视化归一化后的结果
    cmap = 'viridis'
    plt.contourf(x_vals, y_vals, density, cmap=cmap)
    plt.title('Normalized Density Function')
    plt.xlabel('start time (X-axis)')
    plt.ylabel('end time (Y-axis)')
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.colorbar()
    plt.show()
    return density


parser = argparse.ArgumentParser(description="Train model.")
# parser.add_argument("--task", default="charadessta", help="Dataset name in {activitynetcaptions, charadessta, tacos}.")
# parser.add_argument("--task", default="Charades_CD", help="Dataset name in {activitynetcaptions, charadessta, tacos}.")
# parser.add_argument("--task", default="activitynetcaptions", help="Dataset name in {activitynetcaptions, charadessta, tacos}.")
parser.add_argument("--task", default="ANet_CD", help="Dataset name in {activitynetcaptions, charadessta, tacos}.")
parser.add_argument("--biases", type=bool, default=False)
parser.add_argument("--config-file", type=str, default="config_3090.yaml")
args = parser.parse_args()

print("dataset name: {}".format(args.task))

if args.config_file == 'config_3090.yaml':
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
config = load_config("src/{}".format(args.config_file))
config["dataset_name"] = args.task
config["use_biases"] = args.biases
config["config_file"] = args.config_file

data = prepare_data(config, args.task)
train_dl = data["train_dl"]
valid_dl = data["valid_dl"]
test_dl = data["test_dl"]

sample_size = config[args.task]['video_feature_len']

start_idx_bias = []
end_idx_bias = []

for i, batch in tqdm(
    enumerate(train_dl), total=len(train_dl),
    desc="cal_distribution"
):
    # print(batch['start_frame'])
    # print(batch['start_frame'].shape)
    # print(batch['end_frame'])
    # print(batch['end_frame'].shape)
    start_idx_bias.append(batch['start_frame'])
    end_idx_bias.append(batch['end_frame'])
start_idx_bias = np.concatenate(start_idx_bias)
end_idx_bias = np.concatenate(end_idx_bias)

# train_dis = cal_candidate_distribution(
#     coords=np.vstack([start_idx_bias / sample_size, end_idx_bias / sample_size]),
#     sample_size=sample_size,
# )
train_dis_ori = cal_distribution(
    coords=np.vstack([start_idx_bias / sample_size, end_idx_bias / sample_size]),
    sample_size=sample_size,
)

start_idx_bias = []
end_idx_bias = []
for i, batch in tqdm(
    enumerate(test_dl), total=len(test_dl),
    desc="cal_distribution"
):
    # print(batch['start_frame'])
    # print(batch['start_frame'].shape)
    # print(batch['end_frame'])
    # print(batch['end_frame'].shape)
    start_idx_bias.append(batch['start_frame'])
    end_idx_bias.append(batch['end_frame'])

start_idx_bias = np.concatenate(start_idx_bias)
end_idx_bias = np.concatenate(end_idx_bias)

# test_dis = cal_candidate_distribution(
#     coords=np.vstack([start_idx_bias / sample_size, end_idx_bias / sample_size]),
#     sample_size=sample_size,
# )

test_dis_ori = cal_distribution(
    coords=np.vstack([start_idx_bias / sample_size, end_idx_bias / sample_size]),
    sample_size=sample_size,
)

# print(test_dis.shape)
print(test_dis_ori.shape)


moment_length_factors = config[args.task]["moment_length_factors"]
print(moment_length_factors)
overlapping_factors = config[args.task]["overlapping_factors"]
indices = []
video_length = sample_size
# anchor_point = torch.argmax(attn_weight)

for f in moment_length_factors:
    l = round(video_length * f)
    if l == 0:
        continue
    for o in overlapping_factors:
        l_overlap = round(l * o)
        if l == l_overlap:
            continue
        l_rest = l - l_overlap
        # min_index = max(0, anchor_point - l)  # Ablation 3: no anchor point
        # max_index = min(video_length, anchor_point + l)  # Ablation 3: no anchor point
        # starts = range(min_index, anchor_point + 1, l_rest)  # Ablation 3: no anchor point
        # ends = range(min_index + l, max_index + 1, l_rest)  # Ablation 3: no anchor point
        starts = range(0, video_length, l_rest)  # Ablation 3: no anchor point
        ends = range(l, video_length + l, l_rest)  # Ablation 3: no anchor point
        indices.append(torch.stack([torch.tensor([start, end]) for start, end in zip(starts, ends)], dim=0))

indices = torch.cat(indices, dim=0)

# indices[indices >= sample_size] = sample_size - 1
# for i in indices:
#     print(i)
indices = indices[indices[:, 1] < sample_size]


indices = indices.numpy()


candidate_expectation_dis = cal_candidate_distribution(
    coords=np.vstack([indices[:, 0] / sample_size, indices[:, 1] / sample_size]),
    sample_size=sample_size,
)
print(candidate_expectation_dis.shape)
# print(bias_dis)


def kl_divergence(p, q):
    # 转换为numpy数组
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    # 添加一个很小的epsilon值，避免log(0)的问题
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    # 归一化分布
    p = p / np.sum(p)
    q = q / np.sum(q)

    # 计算KL散度
    return np.sum(p * np.log(p / q))


# kld = kl_divergence(test_dis, candidate_expectation_dis)

kld_ori = kl_divergence(test_dis_ori, candidate_expectation_dis)

kld_train = kl_divergence(test_dis_ori, train_dis_ori)

# print(kld)

print("test-candidate", kld_ori)

print("train-test", kld_train)


if __name__ == "__main__":
    import pickle
    with open("b_density_normalized_epoch15.pkl".format(args.task, args.config_file), "rb") as f:
        kld = pickle.load(f)

    kld_train = kl_divergence(kld, train_dis_ori)
    kld_can = kl_divergence(kld, candidate_expectation_dis)

    print("train-candidate", kld_train)
    print("candidate-bias", kld_can)