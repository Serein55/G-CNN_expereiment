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
