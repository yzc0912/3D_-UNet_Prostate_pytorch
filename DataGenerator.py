# datagenerator.py

import glob
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import random
import os
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt

# Function to save random image slices in wandb
def save_sample_images_to_wandb(data, target, epoch):
    # 随机选择一个样本
    sample_idx = random.randint(0, data.shape[0] - 1)
    # 获取数据切片，这里假设每个数据是[batch_size, 1, H, W, D]，我们展示 D 维的切片
    slice_idx = 26
    # 获取真实值和预测值
    gt_slice = target[sample_idx, 0, :, :, slice_idx].cpu().numpy()

    # 可视化切片
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    ax1.imshow(gt_slice, cmap='gray')
    ax1.set_title('Ground Truth')
    plt.tight_layout()
    # 将图像保存到wandb
    # 保存图像文件
    save_path = os.path.join('./', f"epoch_{epoch}_sample_{sample_idx}_slice_{slice_idx}.png")
    fig.savefig(save_path)
    plt.close(fig)  # 关闭图像窗口以节省内存


class CustomDataset(Dataset):
    def __init__(self, data_dir='', transforms=None, train=False):
        """
        初始化数据集，通过列出 CT 图像和标签文件路径来配对。

        Args:
            data_dir (str): 包含 'ct' 和 'label' 子目录的根数据目录（如 'new_dataset/train'）。
            transforms (list, optional): 需要应用的变换列表。
            train (bool): 是否为训练集（启用数据增强）。
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.train = train

        # 定义 CT 图像和标签的目录路径
        self.ct_dir = os.path.join(self.data_dir, 'ct')
        self.label_dir = os.path.join(self.data_dir, 'label')

        # 列出所有 CT 图像文件
        self.image_paths = sorted(glob.glob(os.path.join(self.ct_dir, 'volume-*.nii')))
        # 列出所有标签文件
        self.label_paths = sorted(glob.glob(os.path.join(self.label_dir, 'segmentation-*.nii')))

        # 验证图像和标签数量是否匹配
        assert len(self.image_paths) == len(self.label_paths), "图像和标签数量不匹配"

        # 创建一个映射，从图像编号到标签路径
        self.id_to_label = {}
        for label_path in self.label_paths:
            # 提取编号，例如 'segmentation-12.nii' -> '12'
            basename = os.path.basename(label_path)
            id_num = basename.replace('segmentation-', '').replace('.nii', '')
            self.id_to_label[id_num] = label_path

        # 过滤图像路径，确保每个图像都有对应的标签
        self.filtered_image_paths = []
        self.filtered_label_paths = []
        for image_path in self.image_paths:
            basename = os.path.basename(image_path)
            id_num = basename.replace('volume-', '').replace('.nii', '')
            if id_num in self.id_to_label:
                self.filtered_image_paths.append(image_path)
                self.filtered_label_paths.append(self.id_to_label[id_num])
            else:
                print(f"警告: {image_path} 没有对应的标签文件。")

    def __len__(self):
        return len(self.filtered_image_paths)

    def read_image(self, path):
        """
        使用 SimpleITK 读取 NIfTI 图像。

        Args:
            path (str): NIfTI 文件路径。

        Returns:
            sitk.Image: 读取的图像。
        """
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        return reader.Execute()

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。

        Args:
            idx (int): 数据样本索引。

        Returns:
            tuple: (image_tensor, label_tensor)
        """
        image_path = self.filtered_image_paths[idx]
        label_path = self.filtered_label_paths[idx]

        # 读取图像和标签
        image = self.read_image(image_path)
        label = self.read_image(label_path)

        sample = {'image': image, 'label': label}

        # 应用所有变换
        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        # 将 SimpleITK 图像转换为 NumPy 数组
        image_np = sitk.GetArrayFromImage(sample['image']).astype(np.float32)
        label_np = sitk.GetArrayFromImage(sample['label']).astype(np.int32)

        # 归一化图像强度到 [0, 1]
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-8)

        # 数据增强：随机转置
        if self.train and random.random() > 0.5:
            axes = [2, 1, 0]  # 示例转置轴，根据需要调整
            image_np = image_np.transpose(axes)
            label_np = label_np.transpose(axes)
        else:
            # 默认转置到 [H, W, D]
            image_np = image_np.transpose(1, 2, 0)
            label_np = label_np.transpose(1, 2, 0)

        # 转换为 torch 张量并添加通道维度
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # 形状: [1, H, W, D]
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)  # 形状: [1, H, W, D]

        return image_tensor.float(), label_tensor.long()

# 变换类
class Normalization(object):
    """通过重新缩放强度到 [0, 1] 来归一化图像"""
    def __call__(self, sample):
        resacleFilter = sitk.RescaleIntensityImageFilter()
        resacleFilter.SetOutputMaximum(255)
        resacleFilter.SetOutputMinimum(0)
        image, label = sample['image'], sample['label']
        image = resacleFilter.Execute(image)
        
        return {'image':image, 'label':label}

class RandomFlip(object):
    """随机沿三个轴翻转体积"""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
    
        # generating axis randomly
        flipaxes = np.random.random(3)>0.5
        
        flipimg = sitk.Flip(image, flipaxes.tolist())
        fliplab = sitk.Flip(label, flipaxes.tolist())
        
        return {'image':flipimg, 'label':fliplab}

class RandomSmoothing(object):
    """随机应用高斯平滑"""
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
    
        if np.random.rand() < self.prob:
            image = sitk.RecursiveGaussian(image)
        
        return {'image':image, 'label':label}

class RandomNoise(object):
    """随机添加高斯噪声"""
    def __init__(self, prob, mean=0.0, std=0.05):
        """
        Args:
            prob (float): 添加噪声的概率。
            mean (float): 高斯噪声的均值。
            std (float): 高斯噪声的标准差。
        """
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
    
        if np.random.rand() < self.prob:
            image = sitk.AdditiveGaussianNoise(image)
        
        return {'image':image, 'label':label}

class HistogramMatching(object):
    """使用训练集中的随机图像进行直方图匹配"""
    def __init__(self, train_data_dir, train_size=40, prob=0.5):
        """
        Args:
            train_data_dir (str): 训练集的根数据目录（如 'new_dataset/train'）。
            train_size (int): 用于直方图匹配的训练样本数量。
            prob (float): 应用直方图匹配的概率。
        """
        self.train_size = train_size
        self.prob = prob
        self.files = sorted(glob.glob(os.path.join(train_data_dir, 'ct', 'volume-*.nii')))
        self.files = self.files[:self.train_size]  # 限制为训练集的前 `train_size` 个样本

    def __call__(self, sample):
        if random.random() <= self.prob and len(self.files) > 0:
            index = random.randint(0, len(self.files) - 1)
            template_image_path = self.files[index]
            template_image = sitk.ReadImage(template_image_path)
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(256)
            matcher.SetNumberOfMatchPoints(15)
            matcher.ThresholdAtMeanIntensityOn()
            image = matcher.Execute(sample['image'], template_image)
            return {'image': image, 'label': sample['label']}
        else:
            return sample

# 示例用法
if __name__ == "__main__":
    # 定义训练数据的目录
    train_data_dir = 'new_dataset/train'
    test_data_dir = 'new_dataset/test'

    # 定义变换
    transforms = [
        Normalization(),
        RandomFlip(),
        RandomSmoothing(prob=0.5),
        RandomNoise(prob=0.5, mean=0.0, std=0.05),
        HistogramMatching(train_data_dir=train_data_dir, train_size=40, prob=0.5)
    ]

    # 初始化训练集
    train_dataset = CustomDataset(data_dir=train_data_dir, transforms=transforms, train=True)

    # 初始化测试集（不应用数据增强）
    test_transforms = [
        Normalization()
        # 可以根据需要添加其他变换，但通常测试集不进行数据增强
    ]
    test_dataset = CustomDataset(data_dir=test_data_dir, transforms=test_transforms, train=False)

    # 封装成 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    # 示例: 获取一个训练批次
    for train_image_tensor, train_label_tensor in train_dataloader:
        print(f"训练集批次图像张量形状: {train_image_tensor.shape}")
        print(f"训练集批次标签张量形状: {train_label_tensor.shape}")
        
          # 打印第一批次后退出循环
    save_sample_images_to_wandb(train_image_tensor, train_label_tensor, 0)
    # 示例: 获取一个测试批次
    for test_image_tensor, test_label_tensor in test_dataloader:
        print(f"测试集批次图像张量形状: {test_image_tensor.shape}")
        print(f"测试集批次标签张量形状: {test_label_tensor.shape}")
        break  # 打印第一批次后退出循环
