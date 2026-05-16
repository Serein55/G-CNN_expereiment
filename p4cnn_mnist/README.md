"# P4CNN Rotated MNIST 实验

本实验实现了基于旋转等变神经网络的Rotated MNIST分类任务。

## 目标
- 论文目标错误率：约2.28%
- 使用P4等变卷积神经网络（G-CNN）
- 对称群：p4群（4个离散旋转：0°, 90°, 180°, 270°）

## 文件结构
```
p4cnn_mnist/
├── __init__.py          # 包初始化
├── dataset.py           # Rotated MNIST 数据集加载
├── p4cnn.py            # P4CNN 模型定义
├── C8cnn.py            # P8CNN 模型定义
├── baseline_cnn.py     # Baseline CNN 模型定义
├── main.py             # 统一实验脚本（推荐）
├── train.py            # 训练脚本
├── run_experiment.py    # 一键运行脚本
└── README.md           # 本文件
```

## 环境要求
1. Python 3.6+
2. PyTorch 1.3+
3. torchvision
4. numpy
5. scipy
6. 本项目的escnn库（需要先安装）

## 安装步骤
1. 首先安装escnn库：
```bash
cd /path/to/escnn
pip install -e .
```

2. 安装其他依赖：
```bash
pip install torch torchvision numpy scipy
```

## 快速开始

### 方法1：统一脚本（推荐）
```bash
# 进入实验目录
cd p4cnn_mnist

# 运行 P4CNN 实验（200个epoch）
python main.py --model p4cnn --epochs 200

# 运行 P8CNN 实验
python main.py --model p8cnn --epochs 200

# 运行 Baseline CNN 实验
python main.py --model baseline --epochs 100

# 跳过测试，直接开始训练
python main.py --model p4cnn --skip_tests --epochs 50

# 自定义参数
python main.py --model p4cnn --epochs 200 --batch_size 128 --lr 0.0005
```

### 方法2：使用原有脚本
```bash
# 使用 run_experiment.py
python run_experiment.py --epochs 10

# 使用 train.py
python train.py --model p4cnn --epochs 200
```

### 方法3：分步运行

#### 步骤1：测试数据集加载
```python
from dataset import get_dataloaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
print(f"训练集大小: {len(train_loader.dataset)}")
print(f"验证集大小: {len(val_loader.dataset)}")
print(f"测试集大小: {len(test_loader.dataset)}")
```

#### 步骤2：测试模型定义
```python
import torch
from p4cnn import P4CNN, count_parameters

model = P4CNN(num_classes=10, n_channels=10)
count_parameters(model)

# 测试前向传播
x = torch.randn(4, 1, 28, 28)
output = model(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
```

#### 步骤3：训练模型
```bash
python train.py --epochs 200 --batch_size 64 --lr 0.001
```

## 参数说明

### 模型参数
- `--model`: 模型选择（p4cnn / p8cnn / baseline，默认：p4cnn）
- `--n_channels`: 通道数（p4cnn默认10, p8cnn默认7, baseline默认10）

### 训练参数
- `--epochs`: 训练轮数（默认：200）
- `--batch_size`: 批次大小（默认：64）
- `--lr`: 学习率（默认：0.001）
- `--weight_decay`: 权重衰减（默认：1e-4）
- `--save_dir`: 模型保存目录（默认：./checkpoints）

### 数据集参数
- `--data_dir`: 数据存储目录（默认：./data）
- `--num_workers`: 数据加载线程数（默认：4）

### 其他参数
- `--skip_tests`: 跳过数据集和模型测试

## 模型架构

### P4CNN (P4 等变网络)
P4CNN是一个7层的等变卷积神经网络：

1. **提升层（Lifting Layer）**：
   - 输入：平凡表示（trivial representation）
   - 输出：正则表示（regular representation）
   - 卷积核大小：3×3
   - 作用：将普通图像转换为p4等变特征

2. **中间层（Layer 2-6）**：
   - 输入/输出：正则表示
   - 卷积核大小：3×3
   - 每层包含：卷积 → 批归一化 → ReLU激活
   - 在第2层后添加空间最大池化（2×2）

3. **最后一层（Layer 7）**：
   - 卷积核大小：4×4
   - 作用：进一步提取特征

4. **Group Pooling**：
   - 对旋转维度进行最大池化
   - 获得旋转不变性

5. **分类器**：
   - 全连接层，输出10个类别

### P8CNN (P8 等变网络)
P8CNN是一个6层的等变卷积神经网络，支持45°旋转等变性：

1. **浅层特征提取（Layer 1-2）**：
   - 卷积核大小：5×5（应对45°离散网格混叠）
   - 作用：提取基础特征

2. **深层特征提取（Layer 3-5）**：
   - 卷积核大小：3×3（节省算力）
   - 每层包含：卷积 → 批归一化 → ReLU激活

3. **跨通道混合（Layer 6）**：
   - 卷积核大小：1×1
   - 作用：通道与群混合

4. **自适应池化**：
   - 空间维度：PointwiseAdaptiveAvgPool2D（支持任意输入尺寸）
   - 旋转维度：GroupPooling（获得旋转不变性）

5. **分类器**：
   - 全连接层，输出10个类别

## 预期结果
- 训练时间：约30-60分钟（取决于GPU）
- 最终测试错误率：约2.28%（与论文一致）
- 验证集用于模型选择，测试集用于最终评估

## Checkpoint 目录结构
每个实验的结果保存在 `checkpoints/{model}_n{n_channels}/` 子目录下：

```
checkpoints/
├── p4cnn_n10/
│   ├── best_model.pth           # 最佳模型权重
│   └── training_history.json    # 训练历史记录
├── p8cnn_n10/
│   ├── best_model.pth
│   └── training_history.json
└── baseline_n10/
    ├── best_model.pth
    └── training_history.json
```

- `{model}`: 模型名称（p4cnn / p8cnn / baseline）
- `{n_channels}`: 通道数（默认10）
- 不同参数的实验会自动创建对应的子目录，避免结果覆盖

## 故障排除

### 1. 缺少escnn库
```bash
cd /path/to/escnn
pip install -e .
```

### 2. 内存不足
尝试减小batch_size：
```bash
python train.py --batch_size 32
```

### 3. 训练速度慢
- 检查CUDA是否可用
- 减少num_workers：
```bash
python train.py --num_workers 2
```

### 4. 错误率不收敛
- 尝试调整学习率
- 增加训练轮数
- 检查数据增强是否合理

## 实验原理
P4CNN利用p4群（4个离散旋转）的对称性：
- **等变性**：网络对输入的旋转变换具有等变性
- **权重共享**：通过旋转滤波器实现权重共享，减少参数量
- **不变性**：通过Group Pooling获得旋转不变性

参数量计算：
- P4特征通道数 = 10
- 等效普通CNN通道数 = 10 × √4 = 20
- 与20通道的普通CNN参数量相当

## 参考文献
- Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. In International conference on machine learning (pp. 2990-2999).

## 许可证
本实验代码仅供学术研究使用。

## 联系方式
如有问题，请查看escnn库的文档或提交issue。"