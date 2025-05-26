import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import gaussian_kde


from src.dataset.generate_glance import generate_glance
from src.utils.vl_utils import resample

from src.analysis.cal_kld import train_dis_ori, candidate_expectation_dis, kl_divergence


class BiasDataGenerator(object):
    def __init__(self, dataset_name="Unnamed"):
        self.p_position = None
        self.p = None

        # 只是用于画图
        self.single_clip_size = None
        self.x_vals = None
        self.y_vals = None
        self.b_density_normalized = None
        self.draw_sample = False

        self.cmap = 'viridis'
        self.dataset_name = dataset_name

    def generate_bias_ditributon(self, coords, sample_size, epoch=None, reverse=True):
        show_plot = True

        # 基于当前预测值生成不适的数据分布
        # 替换成你的二维点集a
        # a = np.random.randn(2, 100)  # 用你的实际数据替换，这里示例使用随机生成的二维点集
        # a = np.vstack([x_coords, y_coords])
        a = coords

        # 生成概率密度函数的网格
        x_vals, y_vals = np.meshgrid(
            # np.linspace(a[0].min(), a[0].max(), 100),
            np.linspace(0., 1., sample_size),
            # np.linspace(a[1].min(), a[1].max(), 100)
            np.linspace(0., 1., sample_size)

        )
        positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

        if show_plot and np.unique(a[0]).shape[0] > 1 and np.unique(a[1]).shape[0] > 1:
            # 使用gaussian_kde生成二维概率密度函数
            kde = gaussian_kde(a)
            density = kde(positions).reshape(sample_size, sample_size)

            # 可视化归一化后的结果
            cmap = 'viridis'
            plt.contourf(x_vals, y_vals, density, cmap=cmap)
            plt.title('Normalized Density Function of coord')
            plt.xlabel('start time (X-axis)')
            plt.ylabel('end time (Y-axis)')
            # plt.xlim(0., 1.)
            # plt.ylim(0., 1.)
            plt.colorbar()
            plt.show()

        # print(positions.shape)
        # 根据IoU生成对应候选的权重weight并且用于gaussian_kde

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



        # 使用gaussian_kde生成二维概率密度函数
        # kde = gaussian_kde(a, bw_method=50)
        kde = gaussian_kde(positions, bw_method=None, weights=iou_se.sum(0))


        # density = kde(positions).reshape(100, 100)
        density = kde(positions).reshape(sample_size, sample_size)

        kld_d = kl_divergence(density, candidate_expectation_dis)
        print("KLD between density and candidate_expectation_dis: ", kld_d)
        kld_d_train = kl_divergence(density, train_dis_ori)
        print("KLD between density and train_dis_ori: ", kld_d_train)

        if show_plot:
            # 可视化归一化后的结果
            plt.contourf(x_vals, y_vals, density, cmap=self.cmap)
            plt.title('Normalized {} Density Function of epoch={}'.format(self.dataset_name, epoch))
            plt.xlabel('start time (X-axis)')
            plt.ylabel('end time (Y-axis)')
            plt.xlim(0., 1.)
            plt.ylim(0., 1.)
            plt.colorbar()
            plt.show()

        # 找到概率密度函数的最大值
        max_density = np.max(density)

        b_density = max_density - density

        # 计算b_density，并将x大于y的部分密度改为0
        single_clip_size = 1. / sample_size
        b_density = np.where(x_vals >= y_vals - single_clip_size, 0, b_density)

        # 归一化，确保密度函数的积分等于1
        integral = np.trapz(b_density.flatten())
        b_density_normalized = b_density / integral

        # import pickle
        # with open('b_density_normalized_epoch{}.pkl'.format(epoch), 'wb') as f:
        #     pickle.dump(b_density_normalized, f)


        kld_b = kl_divergence(b_density_normalized, candidate_expectation_dis)
        print("KLD between b and candidate_expectation_dis: ", kld_b)
        kld_b_train = kl_divergence(b_density_normalized, train_dis_ori)
        print("KLD between b and train_dis_ori: ", kld_b_train)

        kld_db = kl_divergence(density, b_density_normalized)
        print("KLD between density and b: ", kld_db)

        if show_plot:
            # 可视化归一化后的结果
            plt.contourf(x_vals, y_vals, b_density_normalized, cmap=self.cmap)
            plt.title('Normalized {} Density Function of epoch={}'.format(self.dataset_name, epoch))
            plt.xlabel('start time (X-axis)')
            plt.ylabel('end time (Y-axis)')
            plt.xlim(0., 1.)
            plt.ylim(0., 1.)
            plt.colorbar()
            plt.show()

        # 再次确保概率密度函数的和为1
        b_normalized = b_density_normalized / np.sum(b_density_normalized)

        self.p_position = positions

        density = np.where(x_vals >= y_vals - single_clip_size, 0, density)
        integral = np.trapz(density.flatten())
        density = density / integral
        self.p = b_normalized if reverse else density

        # 只是用于画图
        self.single_clip_size = single_clip_size
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.b_density_normalized = b_density_normalized
        self.draw_sample = False
        # self.draw_sample = True


    def sample_point(self, batch):
        # 基于不适的数据分布采样一个点或一batch点

        def resample_3times(video_feat: torch.Tensor, target_length: int, cut1: float, cut2: float):
            def resample_wo_padding(vf: torch.Tensor, length: int):
                if vf.shape[0] < length:
                    return vf[np.linspace(start=0, stop=vf.shape[0] - 1, num=length).astype(np.int32)]
                else:
                    return resample(vf, length)[0]

            video_len = video_feat.shape[0]
            ori_cut1 = int(video_len * cut1)
            ori_cut2 = int(video_len * cut2)

            tar_cut1 = int(target_length * cut1)
            tar_cut2 = int(target_length * cut2)

            # print(tar_cut1, tar_cut2, target_length)

            # 如果原始视频三段中的一段长度为0，但target长度不为0，则需要使用这个if补全
            if ori_cut1 == 0:
                f1 = video_feat[[0] * tar_cut1]
            else:
                f1 = resample_wo_padding(video_feat[:ori_cut1], tar_cut1)

            if ori_cut1 == ori_cut2:
                f2 = video_feat[[ori_cut1] * (tar_cut2 - tar_cut1)]
            else:
                f2 = resample_wo_padding(video_feat[ori_cut1:ori_cut2], tar_cut2 - tar_cut1)

            if ori_cut2 == video_len:
                f3 = video_feat[[ori_cut2] * (target_length - tar_cut2)]
            else:
                f3 = resample_wo_padding(video_feat[ori_cut2:], target_length - tar_cut2)

            return torch.cat([f1, f2, f3], dim=0)


        batch_size = batch['video'].shape[0]

        # 从密度函数中生成batch_size个样本
        sampled_indices = np.random.choice(len(self.p_position[0].ravel()), size=batch_size, p=self.p.ravel())
        sampled_points = self.p_position[:, sampled_indices]

        # 确认生成的片段开始时间小于结束时间
        # print(sampled_points.shape)
        # print(sampled_points[:, np.where(sampled_points[0] >= sampled_points[1] - self.single_clip_size)].shape)
        # print(sampled_points[:, np.where(sampled_points[0] < sampled_points[1] - self.single_clip_size)].shape)
        # print(np.where(sampled_points[0] >= sampled_points[1] - self.single_clip_size))
        assert sampled_points[:, np.where(sampled_points[0] >= sampled_points[1] - self.single_clip_size)].shape[2] == 0, \
            sampled_points[:, np.where(sampled_points[0] >= sampled_points[1] - self.single_clip_size)]

        # 对于每个样本单独考虑
        for idx, ov in enumerate(batch['original_video']):
            # print(batch['video'].shape[1])
            # print(batch['start_frac'][idx], batch['end_frac'][idx], batch['glance_frac'][idx])
            # print(sampled_points[0, idx], sampled_points[1, idx])
            # print(ov.shape)

            # 生成新的glance
            new_glance = generate_glance(sampled_points[0, idx], sampled_points[1, idx], precision=4)
            # print(new_glance)

            # 新视频长度为video_feature_len和原长度中的最小值
            # new_target_lenght = min(batch['video'].shape[1], ov.shape[0])
            new_target_lenght = batch['video'].shape[1]
            new_feat = resample_3times(ov, new_target_lenght, sampled_points[0, idx], sampled_points[1, idx])
            # print(new_feat.shape)
            # print()

            # 修改batch内的值
            batch['video'][idx] = new_feat
            batch['video_mask'][idx] = torch.ones(new_feat.shape[0])
            batch['start_frac'][idx], batch['end_frac'][idx], batch['glance_frac'][idx] = \
                sampled_points[0, idx], sampled_points[1, idx], new_glance
            batch['start_frame'][idx], batch['end_frame'][idx], batch['glance_frame'][idx] = \
                (int(sampled_points[0, idx] * new_target_lenght),
                 int(sampled_points[1, idx] * new_target_lenght),
                 int(new_glance * new_target_lenght))

        # 可视化结果
        if self.draw_sample:
            self.draw_sample = False
            plt.scatter(sampled_points[0], sampled_points[1], alpha=0.5, label='Sampled Points')
            plt.contourf(self.x_vals, self.y_vals, self.b_density_normalized, cmap=self.cmap, alpha=0.5)
            plt.title('Generated Samples from Density Function of {}'.format(self.dataset_name))
            plt.xlabel('b values (X-axis)')
            plt.ylabel('b values (Y-axis)')
            plt.colorbar()
            plt.legend()
            plt.show()
        # exit(0)
        return batch



if __name__ == "__main__":
    bg = BiasDataGenerator()

    c = np.random.random((2, 100))
    bg.generate_bias_ditributon(coords=c, sample_size=50, epoch=0)



