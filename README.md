# G-CNN Experiments: 群等变卷积网络的旋转不变性探究

![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c?logo=pytorch&logoColor=white)
![escnn](https://img.shields.io/badge/escnn-Library-blue)
![License](https://img.shields.io/badge/License-MIT-green)

本仓库是论文**《群作用与神经网络输入不变性的讨论——基于群等变卷积网络的研究》**的官方配套代码实现。

本项目主要利用纯代数与群论框架，通过对比标准卷积神经网络 (Baseline CNN) 与基于 `escnn` 构建的群等变卷积网络 (P4-CNN, P8-CNN)，实证探究了模型在存在显著分布偏移（Out-of-Distribution, OOD，即**训练集无旋转，验证集360度随机旋转**）任务下的泛化能力与等变性代数机理。

---

## 📌 核心亮点 / Key Features

1. **数学建模与代码的映射**：利用 `escnn` 库，将连续的 $SE(2)$ 群降维离散化，构建了具有严格代数意义的 $P4 = \mathbb{Z}^2 \rtimes C_4$ 和 $P8$ 等变卷积网络。
2. **严控变量的对比实验**：精确对齐了 Baseline CNN 与 G-CNN 的参数量（均约 19万 参数），排除了模型容量差异带来的干扰。
3. **极端的 OOD 泛化测试**：训练集仅包含平移与水平翻转（绝对不包含旋转），验证集引入 $0^\circ \sim 360^\circ$ 连续随机旋转，直击传统 CNN 的痛点。
4. **插值噪声与过拟合分析**：揭示了离散对称群网络在应对连续旋转插值时引发的分布外过拟合（OOD Overfitting）现象。

---

## 📂 仓库结构 / Repository Structure

```text
G-CNN_experiment/
├── p4cnn_mnist/                # MNIST 相关实验目录 (单通道测试)
│   ├── baseline_cnn.py         # 标准 CNN 基线模型定义
│   ├── p4cnn.py                # P4-CNN (4阶旋转等变) 模型定义
│   ├── p8cnn.py                # P8-CNN (8阶旋转等变) 模型定义
│   ├── dataset.py              # MNIST 旋转数据集构建脚本
│   └── main.py                 # MNIST 训练与测试主入口
│
├── cifar/                      # CIFAR-10 相关实验目录 (三通道复杂特征测试)
│   ├── baseline.py             # CIFAR-10 Baseline CNN 定义
│   ├── model.py                # CIFAR-10 P8-CNN 等变模型定义
│   ├── dataset.py              # CIFAR 数据加载与非对称增强 (验证集360度旋转)
│   └── main.py                 # CIFAR 训练与测试主入口
│
├── data/                       # 数据集自动下载存放目录 (如 MNIST raw)
└── README.md                   # 本说明文档
```
## ⚙️ 环境依赖 / Requirements

建议使用 Python 3.8 及以上版本。运行本项目需要安装以下核心库：

```bash
pip install torch torchvision numpy
pip install escnn   # 群等变卷积核心依赖库
```
(注：escnn 为 e2cnn 的升级版本，支持通用 $E(2)$ 等变可控卷积。)

## 🚀 运行指南 / Usage

1. MNIST 对比实验

进入 p4cnn_mnist 目录并运行主程序。该脚本将自动下载数据集并启动训练：
```bash
cd p4cnn_mnist
python main.py
```
你可以在 main.py 中通过修改实例化的模型（BaselineCNN(), P4CNN(), P8CNN()）来切换网络。

2. CIFAR-10 旋转 OOD 实验 (论文主实验)

进入 cifar 目录，执行测试：
```bash
cd cifar
python main.py
```
该实验将强制在仅包含平移/翻转的 CIFAR-10 训练集上训练网络，并在每一轮验证时施加 360° 随机旋转测试泛化性能。

## 📊 核心实验结果 / Experimental Results

见experiment_results.md

## 🔬 结果分析结论：

CNN 的旋转脆弱性：Baseline CNN 验证集准确率发生崩塌，证明了传统滑动窗口卷积完全不具备旋转等变性。

群池化 (Group Pooling) 的胜利：P8-CNN 利用提升卷积与通道循环移位机制，在数学底层保证了旋转等变，其准确率（57.61%）相较 Baseline 实现了接近一倍的断层式碾压。

工程与代数的碰撞：P8-CNN 验证集未能达到与训练集相同的 87%，且出现了中期过拟合。这主要是因为在离散像素网格上强行施加 $360^\circ$ 连续旋转测试时，不可避免地引入了双线性插值模糊（混叠效应）与截断误差。这种 OOD 噪声突破了离散群等变的理论上限。

## 📚 参考文献 / References

如果你对代码背后的数学原理（特殊欧几里得群 $SE(2)$、晶体学限制定理、等变积分算子）感兴趣，请参考本项目的配套论文。
核心代码的群论实现参考了以下经典工作：

Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. In ICML.

Weiler, M., & Cesa, G. (2019). General E(2)-Equivariant Steerable CNNs. In NeurIPS.

Developed as a final project for Abstract Algebra / Deep Learning studies, May 2026.