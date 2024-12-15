# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time
from model import UNet3D
import DataGenerator as datagenerator
import numpy as np
import argparse
import os
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器
import wandb
import random
import matplotlib.pyplot as plt

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

# Dice loss function
def dice_loss(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

# Function to initialize model weights using Kaiming initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Training function
def train(args):
    wandb.init(project=f'3d-unet-training_{args.learning_rate}_{args.batch_size}', config=args)
    # 超参数从 args 中获取
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    no_of_epochs = args.epochs
    smoothing = args.smoothing
    seed = args.seed
    save_dir = args.save_dir

    # 创建保存模型的目录
    os.makedirs(save_dir, exist_ok=True)

    # 设置随机种子以确保结果可复现
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 定义数据增强和预处理的 transforms
    train_transforms = [
        datagenerator.RandomFlip(),
        datagenerator.HistogramMatching('new_dataset/train/', train_size=40, prob=0.5),
        datagenerator.RandomSmoothing(prob=0.3),
        datagenerator.RandomNoise(prob=0.5),
        datagenerator.Normalization()
    ]
    val_transforms = [datagenerator.Normalization()]

    train_dataset = datagenerator.CustomDataset(
        data_dir='new_dataset/train/',
        transforms=train_transforms,
        train=True
    )

    val_dataset = datagenerator.CustomDataset(
        data_dir='new_dataset/valid/',
        transforms=val_transforms,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型并应用权重初始化
    model = UNet3D()
    initialize_weights(model)  # 应用 Kaiming 初始化，包括 ConvTranspose3d
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

     # 定义学习率调度器
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # 每 100 轮衰减学习率

    # 初始化最佳验证损失为无穷大
    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(1, no_of_epochs + 1):
        start_time = time.time()
        model.train()
        train_losses = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)  # 形状: [batch_size, 1, H, W, D]
            target = target.to(device).float()  # 形状: [batch_size, 1, H, W, D]

            optimizer.zero_grad()
            output = model(data)
            loss = dice_loss(output, target, smooth=smoothing)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            print(f'Epoch: {epoch} - Batch: {batch_idx+1}/{len(train_loader)} - Loss: {np.mean(train_losses):.6f}', end='\r')

        avg_train_loss = np.mean(train_losses)
        elapsed_time = time.time() - start_time
        print(f'\nEpoch: {epoch}/{no_of_epochs} - Train Loss: {avg_train_loss:.6f} - Time: {elapsed_time:.2f}s')

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device).float()
                output = model(data)
                loss = dice_loss(output, target, smooth=smoothing)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f'Epoch: {epoch} - Validation Loss: {avg_val_loss:.6f}')

         # 记录训练和验证损失
        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'epoch': epoch
        })

        # 每20轮保存一个样本切片到wandb
        if epoch % 20 == 0:
            save_sample_images_to_wandb(data, target, output, epoch)

        # 直接与每一轮验证损失比较
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'--> 新的最佳模型已保存，验证损失: {best_val_loss:.6f}')

        # 可选：更新学习率调度器
        scheduler.step()

    print('训练完成。')
    wandb.finish() 

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a 3D U-Net model.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='优化器的学习率。')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='训练的批次大小。')
    parser.add_argument('--epochs', type=int, default=400,
                        help='训练的轮数。')
    parser.add_argument('--smoothing', type=float, default=1e-5,
                        help='Dice 损失的平滑因子。')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='训练数据中用于验证的数据比例。')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子以确保结果可复现。')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='保存模型的目录。')

    args = parser.parse_args()
    train(args)
