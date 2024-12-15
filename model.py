import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split

# ==========================
# 2. Model Architecture
# ==========================

# Define the model
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        self.prelu = nn.PReLU()
        # Encoder layers
        self.enc1 = self.conv_block(1, 16)
        self.down1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.enc2 = self.conv_block(32, 32, num_layers=4)
        self.down2 = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        self.enc3 = self.conv_block(64, 64, num_layers=6)
        self.down3 = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        self.enc4 = self.conv_block(128, 128, num_layers=6)
        self.down4 = nn.Conv3d(128, 256, kernel_size=2, stride=2)
        self.enc5 = self.conv_block(256, 256, num_layers=6)
        # Decoder layers
        self.up1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(256, 256, num_layers=6)
        self.up2 = nn.ConvTranspose3d(256, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 128, num_layers=6)
        self.up3 = nn.ConvTranspose3d(128, 32, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(64, 64, num_layers=4)
        self.up4 = nn.ConvTranspose3d(64, 16, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(32, 32)
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels, num_layers=2):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.PReLU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        res1 = x + enc1
        down1 = self.prelu(self.down1(res1))
        enc2 = self.enc2(down1)
        res2 = down1 + enc2
        down2 = self.prelu(self.down2(res2))
        enc3 = self.enc3(down2)
        res3 = down2 + enc3
        down3 = self.prelu(self.down3(res3))
        enc4 = self.enc4(down3)
        res4 = down3 + enc4
        down4 = self.prelu(self.down4(res4))
        enc5 = self.enc5(down4)
        res5 = down4 + enc5
        # Decoder
        up1 = self.prelu(self.up1(res5))
        concat1 = torch.cat([up1, res4], dim=1)
        dec1 = self.dec1(concat1)
        res6 = concat1 + dec1
        up2 = self.prelu(self.up2(res6))
        concat2 = torch.cat([up2, res3], dim=1)
        dec2 = self.dec2(concat2)
        res7 = concat2 + dec2
        up3 = self.prelu(self.up3(res7))
        concat3 = torch.cat([up3, res2], dim=1)
        dec3 = self.dec3(concat3)
        res8 = concat3 + dec3
        up4 = self.prelu(self.up4(res8))
        concat4 = torch.cat([up4, res1], dim=1)
        dec4 = self.dec4(concat4)
        res9 = concat4 + dec4
        output = self.sigmoid(self.final_conv(res9))
        return output
    
    
if __name__ == '__main__':

    # 创建一个虚拟的输入数据，大小为 (batch_size=1, channels=1, depth=64, height=128, width=128)
    input_tensor = torch.randn(1, 1, 64, 128, 128)  # (B, C, D, H, W)

    # 初始化模型
    model = UNet3D()

    # 将模型设置为评估模式 (关闭 Dropout, BatchNorm 使用训练时的均值等)
    model.eval()

    # 前向传播
    with torch.no_grad():  # 不需要计算梯度
        output = model(input_tensor)

    # 输出结果的形状
    print("Output shape:", output.shape)
