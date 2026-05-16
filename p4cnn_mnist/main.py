#!/usr/bin/env python3
"""
P4CNN Rotated MNIST 统一实验脚本

功能:
1. 检查依赖
2. 测试数据加载
3. 测试模型定义
4. 训练模型 (支持 p4cnn / baseline / 其他)
5. 评估结果

用法:
  python main.py --model p4cnn --epochs 200
  python main.py --model baseline --epochs 100
  python main.py --model p4cnn --skip_tests
"""

import os
import sys
import time
import json
import argparse
import subprocess
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# ==================== 依赖检查 ====================

def check_dependencies() -> bool:
    """检查必要的依赖"""
    print("=" * 60)
    print("检查依赖...")
    print("=" * 60)

    required_packages = ['torch', 'torchvision', 'escnn', 'numpy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n缺少以下包: {', '.join(missing_packages)}")
        print("请使用以下命令安装:")
        print("pip install torch torchvision numpy")
        print("cd /path/to/escnn && pip install -e .")
        return False

    print("\n所有依赖已安装!\n")
    return True


# ==================== 数据集测试 ====================

def test_dataset() -> bool:
    """测试数据集加载"""
    print("=" * 60)
    print("测试数据集加载...")
    print("=" * 60)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dataset import get_dataloaders

        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=100,
            num_workers=2
        )

        # 测试一个批次
        images, labels = next(iter(train_loader))
        print(f"训练集批次形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print("数据集加载测试通过!\n")
        return True

    except Exception as e:
        print(f"数据集加载测试失败: {e}\n")
        return False


# ==================== 模型工厂 ====================

def create_model(model_name: str, num_classes: int = 10, n_channels: int = None, device: str = 'cpu') -> Tuple[nn.Module, dict]:
    """
    根据模型名称创建模型

    Args:
        model_name: 模型名称 (p4cnn / p8cnn / baseline)
        num_classes: 分类数量
        n_channels: 通道数 (默认: p4cnn/baseline=10, p8cnn=7)
        device: 设备
    """
    # 根据模型类型设置默认 n_channels
    if n_channels is None:
        if model_name == 'p8cnn':
            n_channels = 7
        else:
            n_channels = 10

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if model_name == 'p4cnn':
        from p4cnn import P4CNN, count_parameters
        model = P4CNN(num_classes=num_classes, n_channels=n_channels).to(device)
        total_params, trainable_params, equivariant_params = count_parameters(model, verbose=False)
        info = {
            'name': 'P4CNN',
            'n_channels': n_channels,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'equivariant_params': equivariant_params,
        }
        return model, info

    elif model_name == 'p8cnn':
        from p8cnn import P8CNN, count_parameters
        model = P8CNN(num_classes=num_classes, n_channels=n_channels).to(device)
        total_params, trainable_params, equivariant_params = count_parameters(model, verbose=False)
        info = {
            'name': 'P8CNN',
            'n_channels': n_channels,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'equivariant_params': equivariant_params,
        }
        return model, info

    elif model_name == 'baseline':
        from baseline_cnn import make_baseline_matched_to_p4cnn
        from p4cnn import count_parameters
        model, width, width_last, ref_params, baseline_params = make_baseline_matched_to_p4cnn(
            num_classes=num_classes,
            n_channels=n_channels,
        )
        model = model.to(device)
        total_params, trainable_params, equivariant_params = count_parameters(model, verbose=False)
        info = {
            'name': 'BaselineCNN',
            'width': width,
            'width_last': width_last,
            'ref_p4cnn_params': ref_params,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'equivariant_params': equivariant_params,
        }
        return model, info

    else:
        raise ValueError(f"未知的模型名称: {model_name}. 可选: p4cnn, p8cnn, baseline")


def test_model(model_name: str, n_channels: int = 10) -> bool:
    """测试模型定义"""
    print("=" * 60)
    print(f"测试 {model_name} 模型...")
    print("=" * 60)

    try:
        model, info = create_model(model_name, n_channels=n_channels)

        # 测试前向传播
        x = torch.randn(4, 1, 28, 28)
        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"模型: {info['name']}")
        print(f"总参数量: {info['total_params']:,}")
        print(f"等效普通CNN参数量: {info['equivariant_params']:,}")
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print("模型测试通过!\n")
        return True

    except Exception as e:
        print(f"模型测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


# ==================== 训练函数 ====================

def train_one_epoch(model: nn.Module, train_loader, criterion, optimizer, device: str, epoch: int):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'  [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {100. * correct / total:.2f}%')

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    return train_loss, train_acc


def evaluate(model: nn.Module, data_loader, criterion, device: str):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    loss = running_loss / total
    accuracy = 100. * correct / total
    error_rate = 100. - accuracy

    return loss, accuracy, error_rate


def train(args) -> float:
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 生成实验子目录名称: {model}_n{n_channels}
    exp_name = f"{args.model}_n{args.n_channels}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"实验目录: {exp_dir}")

    # 加载数据
    print("\n" + "=" * 60)
    print("加载 Rotated MNIST 数据集...")
    print("=" * 60)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers
    )

    # 创建模型
    print("\n" + "=" * 60)
    print(f"创建 {args.model} 模型...")
    print("=" * 60)
    model, model_info = create_model(
        args.model,
        num_classes=10,
        n_channels=args.n_channels,
        device=device
    )

    # 打印模型信息
    print(f"模型: {model_info['name']}")
    print(f"总参数量: {model_info['total_params']:,}")
    print(f"等效普通CNN参数量: {model_info['equivariant_params']:,}")
    if args.model == 'baseline':
        print(f"Baseline width: {model_info['width']}, width_last: {model_info['width_last']}")
        print(f"参照 P4CNN 参数量: {model_info['ref_p4cnn_params']:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # 训练记录
    best_val_acc = 0.0
    best_val_error = 100.0
    best_epoch = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    print(f"训练配置:")
    print(f"  模型: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    if args.model == 'p4cnn':
        print(f"  P4 通道数 n_channels: {args.n_channels}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 40)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        val_loss, val_acc, val_error = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\n训练 - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Error: {val_error:.2f}%")
        print(f"当前学习率: {current_lr:.6f}")

        # 保存最佳模型 (基于验证集性能)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_error = val_error
            best_epoch = epoch

            # 保存模型
            save_path = os.path.join(exp_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model': args.model,
                'model_info': model_info,
                'n_channels': args.n_channels,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_error': val_error,
                'train_loss': train_loss,
            }, save_path)
            print(f"✓ 保存最佳模型 (验证错误率: {val_error:.2f}%)")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总训练时间: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
    print(f"最佳验证性能 - Epoch {best_epoch}:")
    print(f"  验证准确率: {best_val_acc:.2f}%")
    print(f"  验证错误率: {best_val_error:.2f}%")

    # 加载最佳模型进行测试
    print("\n" + "=" * 60)
    print("在测试集上评估最佳模型...")
    print("=" * 60)

    checkpoint = torch.load(
        os.path.join(exp_dir, 'best_model.pth'),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_error = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\n测试集结果:")
    print(f"  测试准确率: {test_acc:.2f}%")
    print(f"  测试错误率: {test_error:.2f}%")
    print(f"  测试损失: {test_loss:.4f}")

    # 保存训练历史
    history = {
        'model': args.model,
        'model_info': model_info,
        'n_channels': args.n_channels,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_error': test_error,
    }

    history_path = os.path.join(exp_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n训练历史已保存到: {history_path}")

    return test_error


# ==================== 主函数 ====================

def run_full_experiment(args):
    """运行完整实验"""
    print("\n" + "#" * 60)
    print("#  P4CNN Rotated MNIST 实验")
    print("#" * 60 + "\n")

    total_start = time.time()

    # 1. 检查依赖
    if not check_dependencies():
        return

    # 2. 测试数据集
    if not args.skip_tests:
        if not test_dataset():
            return

        # 3. 测试模型
        if not test_model(args.model, args.n_channels):
            return

    # 4. 训练模型
    test_error = train(args)

    total_time = time.time() - total_start

    print("\n" + "#" * 60)
    print(f"#  实验完成!")
    print(f"#  模型: {args.model}")
    print(f"#  测试错误率: {test_error:.2f}%")
    print(f"#  总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")
    print("#" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='P4CNN Rotated MNIST 统一实验脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --model p4cnn --epochs 200
  python main.py --model p8cnn --epochs 200
  python main.py --model baseline --epochs 100
  python main.py --model p4cnn --skip_tests --epochs 50
        """
    )

    # 模型参数
    parser.add_argument(
        '--model', type=str, default='p4cnn', choices=('p4cnn', 'p8cnn', 'baseline'),
        help='模型选择: p4cnn (P4等变) / p8cnn (P8等变) / baseline (传统CNN) (default: p4cnn)'
    )
    parser.add_argument(
        '--n_channels', type=int, default=None,
        help='通道数 (p4cnn默认10, p8cnn默认7, baseline默认10)'
    )

    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数 (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小 (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减 (default: 1e-4)')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据集目录 (default: ./data)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作线程数 (default: 4)')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录 (default: ./checkpoints)')

    # 测试参数
    parser.add_argument('--skip_tests', action='store_true',
                        help='跳过数据集和模型测试')

    args = parser.parse_args()

    # 根据模型类型设置默认 n_channels
    if args.n_channels is None:
        if args.model == 'p8cnn':
            args.n_channels = 7
        else:
            args.n_channels = 10

    # 运行实验
    run_full_experiment(args)


if __name__ == '__main__':
    main()
