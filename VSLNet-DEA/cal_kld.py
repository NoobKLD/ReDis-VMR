import argparse
import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from torch import nn
from tqdm import tqdm

# from generate_bias import BiasDataGenerator
from util.data_gen import gen_or_load_dataset
from util.data_loader_t7 import get_test_loader, get_train_loader
from util.data_util import load_video_features


def cal_candidate_distribution(coords, sample_size, video_lens):
    # 生成概率密度函数的网格
    x_vals, y_vals = np.meshgrid(
        # np.linspace(a[0].min(), a[0].max(), 100),
        np.linspace(0., 1., sample_size),
        # np.linspace(a[1].min(), a[1].max(), 100)
        np.linspace(0., 1., sample_size)

    )
    positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

    iou_mask = np.zeros((sample_size, sample_size)).astype(bool)
    for c in range(coords.shape[1]):
        # iou_mask[int(coords[0, c] * sample_size), int(coords[1, c] * sample_size)] = 1
        iou_mask[int(coords[1, c] * sample_size), int(coords[0, c] * sample_size)] = 1

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

    for idx, video_len in enumerate(video_lens):
        mask = np.zeros((sample_size, sample_size)).astype(bool)
        mask[:video_len, :video_len] = 1
        iou_se[idx, ~mask.flatten()] = 0.

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
    # kde = gaussian_kde(positions, bw_method=5, weights=iou_se.sum(0))

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
    # kde = gaussian_kde(a, bw_method=1.5)
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


parser = argparse.ArgumentParser()
# data parameters
parser.add_argument('--save_dir', type=str, default='datasets_t7', help='path to save processed dataset')
# parser.add_argument('--task', type=str, default='charades', help='target task')
# parser.add_argument('--task', type=str, default='activitynet', help='target task')
parser.add_argument('--task', type=str, default='activitynet_cd', help='target task')
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
configs = parser.parse_args()

sample_size = configs.max_pos_len
# bias_generator = BiasDataGenerator()
print("dataset name: {}".format(configs.task))
dataset = gen_or_load_dataset(configs)
visual_features = load_video_features(os.path.join('data', 'features', configs.task, configs.fv), configs.max_pos_len)
train_loader = get_test_loader(dataset=dataset['train_set'], video_features=visual_features, configs=configs)
test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, configs=configs)


video_features = dict()
filenames = glob.glob(os.path.join(os.path.join('data', 'features', configs.task, configs.fv), "*.npy"))
for filename in tqdm(filenames, total=len(filenames), desc="load video shape"):
    video_id = filename.split("/")[-1].split(".")[0]
    video_features[video_id] = np.load(filename).shape[0]

start_idx_bias = []
end_idx_bias = []
for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc='cal_distribution'):
    for record in records:
        s_idx = record["s_time"] / record["duration"] * min(video_features[record['vid']], sample_size)
        e_idx = record["e_time"] / record["duration"] * min(video_features[record['vid']], sample_size)
        start_idx_bias.append(s_idx)
        end_idx_bias.append(e_idx)
start_idx_bias = np.array(start_idx_bias).astype(np.int32)
end_idx_bias = np.array(end_idx_bias).astype(np.int32)
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
for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
        enumerate(test_loader), total=len(test_loader), desc='cal_distribution'):
    for record in records:
        s_idx = record["s_time"] / record["duration"] * min(video_features[record['vid']], sample_size)
        e_idx = record["e_time"] / record["duration"] * min(video_features[record['vid']], sample_size)
        start_idx_bias.append(s_idx)
        end_idx_bias.append(e_idx)

start_idx_bias = np.array(start_idx_bias).astype(np.int32)
end_idx_bias = np.array(end_idx_bias).astype(np.int32)

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
# print(test_dis)



start_idx_bias = []
end_idx_bias = []
video_lens = []
for i in range(sample_size):
    for j in range(sample_size):
        for video_len in range(sample_size):
            if video_len % 16 != 0:   # for fast calculation
                continue
            if i < j <= video_len:
                start_idx_bias.append(i)
                end_idx_bias.append(j)
                video_lens.append(video_len)

start_idx_bias = np.array(start_idx_bias).astype(np.int32)
end_idx_bias = np.array(end_idx_bias).astype(np.int32)
video_lens = np.array(video_lens).astype(np.int32)

candidate_expectation_dis = cal_candidate_distribution(
    coords=np.vstack([start_idx_bias / sample_size, end_idx_bias / sample_size]),
    sample_size=sample_size,
    video_lens=video_lens
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

    # charades_path = '/media/kld/code/VSLNet-master/b_density_normalized_epoch29.pkl'
    # with open(charades_path, 'rb') as f:
    #     charades_data = pickle.load(f)
    #
    # print("redis_charades-candidate", kl_divergence(charades_data, candidate_expectation_dis))
    # print("redis_charades-train", kl_divergence(charades_data, train_dis_ori))

    activitynet_path = '/media/kld/code/VSLNet-master/b_density_normalized_epoch49.pkl'
    with open(activitynet_path, 'rb') as f:
        activitynet_data = pickle.load(f)
        activitynet_data = activitynet_data + test_dis_ori
        activitynet_data = activitynet_data / activitynet_data.sum()

    print("redis_activitynet-candidate", kl_divergence(activitynet_data, candidate_expectation_dis))
    print("redis_activitynet-train", kl_divergence(activitynet_data, train_dis_ori))

    x_vals, y_vals = np.meshgrid(
        # np.linspace(a[0].min(), a[0].max(), 100),
        np.linspace(0., 1., sample_size),
        # np.linspace(a[1].min(), a[1].max(), 100)
        np.linspace(0., 1., sample_size)

    )

    # 可视化归一化后的结果
    cmap = 'viridis'
    plt.contourf(x_vals, y_vals, activitynet_data, cmap=cmap)
    plt.title('Normalized candidate Density Function')
    plt.xlabel('start time (X-axis)')
    plt.ylabel('end time (Y-axis)')
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)
    plt.colorbar()
    plt.show()
