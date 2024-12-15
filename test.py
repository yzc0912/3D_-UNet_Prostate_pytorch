import torch
import torch.nn as nn
import os
import time
import numpy as np
import wandb
from model import UNet3D
import DataGenerator as datagenerator
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import argparse
from metric_eval import jc, dc, hd, hd95
from skimage.measure import find_contours
"""
# 保存样本图像到wandb
def save_sample_images_to_wandb(data, target, output, epoch):
    # 随机选择一个样本
    sample_idx = random.randint(0, data.shape[0] - 1)
    # 获取数据切片，这里假设每个数据是[batch_size, 1, H, W, D]，我们展示 D 维的切片
    slice_idx = data.shape[4] // 2

    # 获取真实值和预测值
    gt_slice = target[sample_idx, 0, :, :, slice_idx].cpu().numpy()
    pred_slice = output[sample_idx, 0, :, :, slice_idx].cpu().detach().numpy()

    # 可视化切片
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(gt_slice, cmap='gray')
    ax1.set_title('Ground Truth')
    ax2.imshow(pred_slice, cmap='gray')
    ax2.set_title('Prediction')
    plt.tight_layout()

    # 将图像保存到wandb
    wandb.log({f'epoch_{epoch}_sample_image': [wandb.Image(fig)]})
    plt.close(fig)
"""

# 保存样本图像到wandb
def save_sample_images_to_wandb(data, target, output, epoch):
    # 随机选择一个样本
    sample_idx = random.randint(0, data.shape[0] - 1)
    # 获取数据切片，这里假设每个数据是[batch_size, 1, H, W, D]，我们展示 D 维的切片
    slice_idx = data.shape[4] // 2

    # 获取真实值和预测值
    gt_slice = target[sample_idx, 0, :, :, slice_idx].cpu().numpy()
    pred_slice = output[sample_idx, 0, :, :, slice_idx].cpu().detach().numpy()

    # 将预测值和真实值二值化
    pred_binary = (pred_slice > 0.5).astype(np.uint8)  # 假设阈值为0.5
    gt_binary = (gt_slice > 0.5).astype(np.uint8)  # 假设阈值为0.5

    # 找到真实和预测分割区域的边界
    pred_contours = find_contours(pred_binary, 0.5)
    gt_contours = find_contours(gt_binary, 0.5)

    # 可视化切片
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.ones_like(gt_slice), cmap='gray')  # 背景为白色

    # 绘制预测的红色边界
    for contour in pred_contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', lw=2, label='Prediction' if 'Prediction' not in [line.get_label() for line in ax.lines] else "")  # 添加label

    # 绘制真实的绿色边界
    for contour in gt_contours:
        ax.plot(contour[:, 1], contour[:, 0], color='green', lw=2, label='Ground Truth' if 'Ground Truth' not in [line.get_label() for line in ax.lines] else "")  # 添加label

    # 添加图例
    ax.legend(loc='upper right')

    ax.set_title(f'Epoch {epoch} - Sample {sample_idx}')
    plt.tight_layout()

    # 将图像保存到wandb
    wandb.log({f'epoch_{epoch}_sample_image': [wandb.Image(fig)]})
    plt.close(fig)

# Dice loss function
def dice_loss(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))


# Testing function
def test(args):
    wandb.init(project=f'3d-unet-testing_{args.learning_rate}_{args.batch_size}', config=args)
    # 超参数从 args 中获取
    batch_size = 1
    model_path = args.model_path

    # 定义数据预处理的 transforms
    test_transforms = [datagenerator.Normalization()]

    test_dataset = datagenerator.CustomDataset(
        data_dir='new_dataset/test/',
        transforms=test_transforms,
        train=False
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型并应用权重初始化
    model = UNet3D()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 加载训练好的模型
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 初始化损失和指标
    test_losses = []
    dc_all = []
    jc_all = []
    hd_all = []
    hd95_all = []

    # 用来保存每个batch的所有指标
    batch_results = []

    # 测试循环
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)  # 形状: [batch_size, 1, H, W, D]
        target = target.to(device).float()  # 形状: [batch_size, 1, H, W, D]

        # 前向传播
        with torch.no_grad():
            output = model(data)
            loss = dice_loss(output, target)
            test_losses.append(loss.item())
            
            # 转换成二值图像：你可以选择合适的阈值
            output_bin = output > 0.5  # 假设0.5是合适的阈值
            target_bin = target > 0.5  # 假设目标也需要二值化

            # 计算指标
            dc_batch = dc(output_bin.cpu().numpy(), target_bin.cpu().numpy())
            jc_batch = jc(output_bin.cpu().numpy(), target_bin.cpu().numpy())
            hd_batch = hd(output_bin.cpu().numpy(), target_bin.cpu().numpy())
            hd95_batch = hd95(output_bin.cpu().numpy(), target_bin.cpu().numpy())
            
            # 保存每个batch的指标
            batch_results.append({
                'batch_idx': batch_idx,
                'dice_coefficient': dc_batch,
                'jaccard_index': jc_batch,
                'hausdorff_distance': hd_batch,
                'hausdorff_distance_95': hd95_batch
            })

            dc_all.append(dc_batch)
            jc_all.append(jc_batch)
            hd_all.append(hd_batch)
            hd95_all.append(hd95_batch)

            # 每个batch都保存一个样本的预测结果
            save_sample_images_to_wandb(data, target, output, epoch=batch_idx)

        print(f'Batch: {batch_idx+1}/{len(test_loader)} - Loss: {np.mean(test_losses):.6f}', end='\r')

    avg_test_loss = np.mean(test_losses)
    avg_dc = np.mean(dc_all)
    avg_jc = np.mean(jc_all)
    avg_hd = np.mean(hd_all)
    avg_hd95 = np.mean(hd95_all)

    # 打印测试结果
    print(f'\nTest Loss: {avg_test_loss:.6f}')
    print(f'Dice Coefficient: {avg_dc:.6f}')
    print(f'Jaccard Index: {avg_jc:.6f}')
    print(f'Hausdorff Distance: {avg_hd:.6f}')
    print(f'95th Percentile Hausdorff Distance: {avg_hd95:.6f}')

    # 记录所有的测试指标到wandb
    for result in batch_results:
        wandb.log({
            'batch_idx': result['batch_idx'],
            'dice_coefficient': result['dice_coefficient'],
            'jaccard_index': result['jaccard_index'],
            'hausdorff_distance': result['hausdorff_distance'],
            'hausdorff_distance_95': result['hausdorff_distance_95']
        })

    # 总体平均指标记录
    wandb.log({
        'avg_test_loss': avg_test_loss,
        'avg_dice_coefficient': avg_dc,
        'avg_jaccard_index': avg_jc,
        'avg_hausdorff_distance': avg_hd,
        'avg_hausdorff_distance_95': avg_hd95
    })

    print('测试完成。')
    wandb.finish()

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Test a 3D U-Net model.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='优化器的学习率。')
    parser.add_argument('--batch_size', type=int, default=1, help='测试的批次大小。')
    parser.add_argument('--model_path', type=str, default='saved_models/best_model.pth', help='加载的训练好的模型路径。')

    args = parser.parse_args()
    test(args)
