# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# 2d conv blocks
class Basic2DBlock(nn.Module):
    """
    基础2D卷积块：包含卷积层、批归一化和ReLU激活
    
    结构: Conv2D -> BatchNorm -> ReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size):
        """
        初始化基础2D卷积块
        
        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
            kernel_size: 卷积核大小
        """
        super(Basic2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.block(x)


class Res2DBlock(nn.Module):
    """
    2D残差块：包含两个卷积层和一个跳跃连接
    
    主分支: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm
    跳跃连接: 如果输入输出通道数不同，使用1x1卷积调整维度
    最后输出: ReLU(主分支 + 跳跃连接)
    """
    def __init__(self, in_planes, out_planes):
        """
        初始化2D残差块
        
        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
        """
        super(Res2DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes)
        )

        if in_planes == out_planes: 
            self.skip_con = nn.Sequential()  # 如果通道数相同，使用恒等映射
        else:
            self.skip_con = nn.Sequential(  # 如果通道数不同，使用1x1卷积调整维度
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes)
            )
    
    def forward(self, x):
        """前向传播"""
        res = self.res_branch(x)
        skip = self.skip_con(x)  # 跳跃连接
        return F.relu(res + skip, True)

    
class Pool2DBlock(nn.Module):
    """
    2D池化块：使用最大池化进行下采样
    """
    def __init__(self, pool_size):
        """
        初始化2D池化块
        
        Args:
            pool_size: 池化窗口大小和步长
        """
        super(Pool2DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        """前向传播"""
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Upsample2DBlock(nn.Module):
    """
    2D上采样块：使用反卷积进行上采样
    
    结构: ConvTranspose2D -> BatchNorm -> ReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        """
        初始化2D上采样块
        
        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
            kernel_size: 卷积核大小(固定为2)
            stride: 步长(固定为2)
        """
        super(Upsample2DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        """前向传播"""
        return self.block(x)


class EncoderDecorder(nn.Module):
    """
    编码器-解码器结构：用于特征提取和语义分割
    
    编码器部分: 两次下采样，逐渐增加通道数
    解码器部分: 两次上采样，逐渐减少通道数
    包含跳跃连接，结合低层特征和高层特征
    """
    def __init__(self):
        """初始化编码器-解码器结构"""
        super(EncoderDecorder, self).__init__()

        # 编码器部分
        self.encoder_pool1 = Pool2DBlock(2)  # 第一次下采样，空间尺寸减半
        self.encoder_res1 = Res2DBlock(32, 64)  # 通道数增加到64
        self.encoder_pool2 = Pool2DBlock(2)  # 第二次下采样，空间尺寸再减半
        self.encoder_res2 = Res2DBlock(64, 128)  # 通道数增加到128

        # 中间部分
        self.mid_res = Res2DBlock(128, 128)  # 保持通道数不变

        # 解码器部分
        self.decoder_res2 = Res2DBlock(128, 128)  # 保持通道数不变
        self.decoder_upsample2 = Upsample2DBlock(128, 64, 2, 2)  # 第一次上采样，通道数减少到64
        self.decoder_res1 = Res2DBlock(64, 64)  # 保持通道数不变
        self.decoder_upsample1 = Upsample2DBlock(64, 32, 2, 2)  # 第二次上采样，通道数减少到32

        # 跳跃连接部分
        self.skip_res1 = Res2DBlock(32, 32)  # 处理第一层特征
        self.skip_res2 = Res2DBlock(64, 64)  # 处理第二层特征

    def forward(self, x):
        """前向传播"""
        # 编码器前向传播，同时保存中间特征用于跳跃连接
        skip_x1 = self.skip_res1(x)  # 第一层特征
        x = self.encoder_pool1(x)  # 第一次下采样
        x = self.encoder_res1(x)  # 特征提取

        skip_x2 = self.skip_res2(x)  # 第二层特征
        x = self.encoder_pool2(x)  # 第二次下采样
        x = self.encoder_res2(x)  # 特征提取

        # 中间部分
        x = self.mid_res(x)

        # 解码器前向传播，结合跳跃连接
        x = self.decoder_res2(x)  # 特征处理
        x = self.decoder_upsample2(x)  # 第一次上采样
        x = x + skip_x2  # 添加第二层跳跃连接

        x = self.decoder_res1(x)  # 特征处理
        x = self.decoder_upsample1(x)  # 第二次上采样
        x = x + skip_x1  # 添加第一层跳跃连接

        return x


class P2PNet(nn.Module):
    """
    点到点网络(Point-to-Point Network)
    
    用于从体素特征生成热图的网络，包含前端卷积层、编码器-解码器和输出层
    """
    def __init__(self, input_channels, output_channels):
        """
        初始化P2P网络
        
        Args:
            input_channels: 输入通道数
            output_channels: 输出通道数
        """
        super(P2PNet, self).__init__()
        self.output_channels = output_channels

        # 前端卷积层
        self.front_layers = nn.Sequential(
            Basic2DBlock(input_channels, 16, 7),  # 7x7卷积提取初始特征
            Res2DBlock(16, 32),  # 残差块进一步处理特征
        )

        # 编码器-解码器结构
        self.encoder_decoder = EncoderDecorder()

        # 输出层
        self.output_layer = nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        """前向传播"""
        x = self.front_layers(x)  # 前端特征提取
        x = self.encoder_decoder(x)  # 编码-解码特征处理
        x = self.output_layer(x)  # 输出层生成热图
        return x

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


class CenterNet(nn.Module):
    """
    中心点检测网络
    
    用于检测人体中心点位置的网络，同时输出中心点热图和边界框大小
    与P2PNet类似，但有两个输出分支：热图分支和大小分支
    """
    def __init__(self, input_channels, output_channels, head_conv=32):
        """
        初始化CenterNet
        
        Args:
            input_channels: 输入通道数
            output_channels: 输出通道数(热图的类别数)
            head_conv: 输出头部的中间通道数
        """
        super(CenterNet, self).__init__()
        self.output_channels = output_channels

        # 前端卷积层
        self.front_layers = nn.Sequential(
            Basic2DBlock(input_channels, 16, 7),  # 7x7卷积提取初始特征
            Res2DBlock(16, 32),  # 残差块进一步处理特征
        )

        # 编码器-解码器结构
        self.encoder_decoder = EncoderDecorder()

        # 热图输出分支
        self.output_hm = nn.Sequential(
            nn.Conv2d(32, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, output_channels, kernel_size=1, padding=0)
        )

        # 大小输出分支(预测宽高)
        self.output_size = nn.Sequential(
            nn.Conv2d(32, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, padding=0, bias=True)  # 输出2个通道：宽和高
        )
        
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入的3D特征体素，形状为[batch_size, input_channels, depth, height, width]
            
        Returns:
            hm: 输出的2D热图，形状为[batch_size, output_channels, height, width]
            size: 输出的2D大小预测，形状为[batch_size, 2, height, width]
        """
        x, _ = torch.max(x, dim=4)  # 沿Z轴最大池化，从3D压缩到2D
        x = self.front_layers(x)  # 前端特征提取
        x = self.encoder_decoder(x)  # 编码-解码特征处理
        hm, size = self.output_hm(x), self.output_size(x)  # 分别生成热图和大小预测
        return hm, size

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)