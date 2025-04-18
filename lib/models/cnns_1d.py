# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class Basic1DBlock(nn.Module):
    """
    基础1D卷积块：包含卷积层、批归一化和ReLU激活
    
    结构: Conv1D -> BatchNorm -> ReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size):
        """
        初始化基础1D卷积块
        
        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
            kernel_size: 卷积核大小
        """
        super(Basic1DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.block(x)


class Res1DBlock(nn.Module):
    """
    1D残差块：包含两个卷积层和一个跳跃连接
    
    主分支: Conv1D -> BatchNorm -> ReLU -> Conv1D -> BatchNorm
    跳跃连接: 如果输入输出通道数不同，使用1x1卷积调整维度
    最后输出: ReLU(主分支 + 跳跃连接)
    """
    def __init__(self, in_planes, out_planes):
        """
        初始化1D残差块
        
        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
        """
        super(Res1DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True),
            nn.Conv1d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_planes)
        )

        if in_planes == out_planes: 
            self.skip_con = nn.Sequential()  # 如果通道数相同，使用恒等映射
        else:
            self.skip_con = nn.Sequential(  # 如果通道数不同，使用1x1卷积调整维度
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_planes)
            )
    
    def forward(self, x):
        """前向传播"""
        res = self.res_branch(x)
        skip = self.skip_con(x)  # 跳跃连接
        return F.relu(res + skip, True)

    
class Pool1DBlock(nn.Module):
    """
    1D池化块：使用最大池化进行下采样
    """
    def __init__(self, pool_size):
        """
        初始化1D池化块
        
        Args:
            pool_size: 池化窗口大小和步长
        """
        super(Pool1DBlock, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        """前向传播"""
        return F.max_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
    

class Upsample1DBlock(nn.Module):
    """
    1D上采样块：使用反卷积进行上采样
    
    结构: ConvTranspose1D -> BatchNorm -> ReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        """
        初始化1D上采样块
        
        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
            kernel_size: 卷积核大小(固定为2)
            stride: 步长(固定为2)
        """
        super(Upsample1DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        """前向传播"""
        return self.block(x)

class EncoderDecorder(nn.Module):
    """
    编码器-解码器结构：用于1D特征提取和处理
    
    编码器部分: 两次下采样，逐渐增加通道数
    解码器部分: 两次上采样，逐渐减少通道数
    包含跳跃连接，结合低层特征和高层特征
    """
    def __init__(self):
        """初始化编码器-解码器结构"""
        super(EncoderDecorder, self).__init__()

        # 编码器部分
        self.encoder_pool1 = Pool1DBlock(2)  # 第一次下采样，序列长度减半
        self.encoder_res1 = Res1DBlock(32, 64)  # 通道数增加到64
        self.encoder_pool2 = Pool1DBlock(2)  # 第二次下采样，序列长度再减半
        self.encoder_res2 = Res1DBlock(64, 128)  # 通道数增加到128

        # 中间部分
        self.mid_res = Res1DBlock(128, 128)  # 保持通道数不变

        # 解码器部分
        self.decoder_res2 = Res1DBlock(128, 128)  # 保持通道数不变
        self.decoder_upsample2 = Upsample1DBlock(128, 64, 2, 2)  # 第一次上采样，通道数减少到64
        self.decoder_res1 = Res1DBlock(64, 64)  # 保持通道数不变
        self.decoder_upsample1 = Upsample1DBlock(64, 32, 2, 2)  # 第二次上采样，通道数减少到32

        # 跳跃连接部分
        self.skip_res1 = Res1DBlock(32, 32)  # 处理第一层特征
        self.skip_res2 = Res1DBlock(64, 64)  # 处理第二层特征

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


class C2CNet(nn.Module):
    """
    高度检测网络(Column to Column Network)
    
    用于检测人体的高度(z坐标)，处理1D特征序列
    接收从2D特征图中提取的1D柱状特征，预测每个位置的高度概率
    """
    def __init__(self, input_channels, output_channels, head_conv=32):
        """
        初始化C2C网络
        
        Args:
            input_channels: 输入通道数
            output_channels: 输出通道数
            head_conv: 输出头部的中间通道数(未使用)
        """
        super(C2CNet, self).__init__()
        self.output_channels = output_channels

        # 前端卷积层
        self.front_layers = nn.Sequential(
            Basic1DBlock(input_channels, 16, 7),  # 7x1卷积提取初始特征
            Res1DBlock(16, 32),  # 残差块进一步处理特征
        )

        # 编码器-解码器结构
        self.encoder_decoder = EncoderDecorder()

        # 输出层：生成1D高度热图
        self.output_hm = nn.Conv1d(32, output_channels, kernel_size=1, stride=1, padding=0)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入的1D特征，形状为[batch_size, input_channels, depth]
            
        Returns:
            hm: 输出的1D高度热图，形状为[batch_size, output_channels, depth]
        """
        x = self.front_layers(x)  # 前端特征提取
        x = self.encoder_decoder(x)  # 编码-解码特征处理
        hm = self.output_hm(x)  # 输出高度热图
        return hm

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose1d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
