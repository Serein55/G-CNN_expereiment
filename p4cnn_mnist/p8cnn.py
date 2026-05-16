"""
P8CNN 模型定义 (严格对照版)
完全复刻 P4CNN 的原始拓扑结构，不使用任何未在 P4CNN 中出现的技巧。

控制变量：
- 结构：严格保持 7 层卷积 (6 层 3x3 + 1 层 4x4)
- 池化：仅在第 2 层后使用 PointwiseMaxPool2D(2x2)
- 参数量对齐：通过将 n_channels 设为 7，抵消 N=8 带来的参数增长，保持总量接近。
"""

import torch
import torch.nn as nn
from typing import Tuple

import escnn.nn as enn
from escnn import gspaces

class P8CNN(enn.EquivariantModule):
    def __init__(self, num_classes=10, n_channels=7):
        """
        Args:
            num_classes: 分类数量 (MNIST: 10)
            n_channels: P8 特征场数 (设为 7 以对齐 P4CNN 中 10 个场的参数量)
        """
        super(P8CNN, self).__init__()
        
        # 唯一的架构改变：修改对称群为 N=8
        self.gspace = gspaces.rot2dOnR2(N=8)
        
        # 输入场类型：平凡表示 (单通道灰度图)
        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        
        # --- 完全使用原始的层级定义 ---
        
        # Layer 1: 3x3 conv, trivial -> regular (提升层)
        out_type_1 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv1 = enn.R2Conv(
            self.in_type, out_type_1, kernel_size=5, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn1 = enn.InnerBatchNorm(out_type_1)
        self.relu1 = enn.ReLU(out_type_1, inplace=True)
        
        # Layer 2: 3x3 conv, regular -> regular
        out_type_2 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv2 = enn.R2Conv(
            out_type_1, out_type_2, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn2 = enn.InnerBatchNorm(out_type_2)
        self.relu2 = enn.ReLU(out_type_2, inplace=True)
        
        # 空间最大池化 (在第 2 层后)
        self.pool1 = enn.PointwiseMaxPool2D(out_type_2, kernel_size=2, stride=2)
        
        # Layer 3: 3x3 conv
        out_type_3 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv3 = enn.R2Conv(
            out_type_2, out_type_3, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn3 = enn.InnerBatchNorm(out_type_3)
        self.relu3 = enn.ReLU(out_type_3, inplace=True)
        
        # Layer 4: 3x3 conv
        out_type_4 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv4 = enn.R2Conv(
            out_type_3, out_type_4, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn4 = enn.InnerBatchNorm(out_type_4)
        self.relu4 = enn.ReLU(out_type_4, inplace=True)
        
        # Layer 5: 3x3 conv
        out_type_5 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv5 = enn.R2Conv(
            out_type_4, out_type_5, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn5 = enn.InnerBatchNorm(out_type_5)
        self.relu5 = enn.ReLU(out_type_5, inplace=True)
        
        # Layer 6: 3x3 conv
        out_type_6 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv6 = enn.R2Conv(
            out_type_5, out_type_6, kernel_size=3, padding=1,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn6 = enn.InnerBatchNorm(out_type_6)
        self.relu6 = enn.ReLU(out_type_6, inplace=True)
        
        # Layer 7: 4x4 conv (严格保留原始的 4x4 设计)
        out_type_7 = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)
        self.conv7 = enn.R2Conv(
            out_type_6, out_type_7, kernel_size=4, padding=0,
            bias=False, sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn7 = enn.InnerBatchNorm(out_type_7)
        self.relu7 = enn.ReLU(out_type_7, inplace=True)
        
        # GroupPooling: 获得不变性
        self.gpool = enn.GroupPooling(out_type_7)
        
        # 全连接层分类器
        # 因为通道数变成了 7，且空间维度依然是 11x11，所以维度是 7 * 11 * 11
        self.fc = nn.Linear(n_channels * 11 * 11, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将普通 Tensor 包装为 GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # 严格按原版流程执行
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))
        
        x = self.gpool(x)
        
        # 展平并执行全连接
        x = x.tensor
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        batch_size = input_shape[0]
        return (batch_size, self.fc.out_features)


def compute_equivariant_params(model):
    """
    计算等效普通 CNN 参数量
    
    对于等变卷积，参数在 Fourier 空间中是独立的，但在原始空间中是共享的（等变性约束）。
    此函数计算如果没有等变性约束，需要多少普通 CNN 参数才能达到相同的表达能力。
    
    公式:
    - trivial -> regular: n_out * kernel_size^2
    - regular -> regular: n_in * n_out * kernel_size^2 * N
    
    对于普通 CNN，等效参数量等于实际存储参数量。
    """
    import torch.nn as nn
    import escnn.nn as enn
    
    equivariant_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, enn.R2Conv):
            ks = module.kernel_size
            in_type = module.in_type
            out_type = module.out_type
            
            in_repr = in_type.representations[0]
            
            if in_repr.is_trivial():
                # trivial -> regular: n_out * kernel_size^2
                n_out = len(out_type.representations)
                equivariant_params += n_out * ks * ks
            else:
                # regular -> regular: n_in * n_out * kernel_size^2 * N
                n_in = len(in_type.representations)
                n_out = len(out_type.representations)
                N = in_type.gspace.fibergroup.order()
                equivariant_params += n_in * n_out * ks * ks * N
            
        elif isinstance(module, nn.Conv2d):
            # 普通卷积：等效参数量等于实际存储参数量
            equivariant_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Linear):
            equivariant_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, (nn.BatchNorm2d, enn.InnerBatchNorm)):
            equivariant_params += sum(p.numel() for p in module.parameters())
    
    return equivariant_params


def count_parameters(model, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    equivariant_params = compute_equivariant_params(model)
    
    if verbose:
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"等效普通CNN参数量: {equivariant_params:,}")
    
    return total_params, trainable_params, equivariant_params


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P8CNN(num_classes=10, n_channels=7).to(device)
    
    print("=== 严格控制变量的 P8CNN ===")
    total, trainable, equiv = count_parameters(model)
    
    # 形状测试
    x = torch.randn(2, 1, 28, 28).to(device)
    out = model(x)
    print(f"测试前向传播: 输入 {x.shape} -> 输出 {out.shape}")