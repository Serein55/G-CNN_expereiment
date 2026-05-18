import os
import sys
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_model(model_name='p8cnn', num_classes=10, n_channels=32, device='cpu'):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if model_name == 'p8cnn':
        from model import CIFAR_P8CNN, count_parameters
        model = CIFAR_P8CNN(num_classes=num_classes, n_channels=n_channels).to(device)
        total_params, trainable_params, _ = count_parameters(model, verbose=False)
        info = {
            'name': 'CIFAR_P8CNN',
            'n_channels': n_channels,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
    elif model_name == 'baseline':
        from baseline import BaselineCNN, make_baseline_matched_to_p8cnn, count_parameters
        model, width, ref_params, baseline_params = make_baseline_matched_to_p8cnn(
            num_classes=num_classes, n_channels=n_channels,
        )
        model = model.to(device)
        total_params, trainable_params = count_parameters(model, verbose=False)
        info = {
            'name': 'BaselineCNN',
            'width': width,
            'ref_p8cnn_params': ref_params,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, info


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
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

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'  [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {100. * correct / total:.2f}%')

    train_loss = running_loss / total
    train_acc = 100. * correct / total
    return train_loss, train_acc


def evaluate(model, data_loader, criterion, device):
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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_name = f"cifar_{args.model}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment dir: {exp_dir}")

    print("\nLoading CIFAR-10 dataset...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import get_dataloaders
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )

    print("\nCreating model...")
    model, model_info = create_model(
        model_name=args.model,
        num_classes=10,
        n_channels=args.n_channels,
        device=device,
    )
    print(f"Model: {model_info['name']}")
    print(f"Total params: {model_info['total_params']:,}")
    if args.model == 'baseline':
        print(f"Baseline width: {model_info['width']}")
        print(f"Reference P8CNN params: {model_info['ref_p8cnn_params']:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_test_acc = 0.0
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
        )

        test_loss, test_acc, test_error = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nTrain - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Test  - Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | Error: {test_error:.2f}%")
        print(f"LR: {current_lr:.6f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            save_path = os.path.join(exp_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'test_error': test_error,
            }, save_path)
            print(f"Saved best model (error: {test_error:.2f}%)")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"Training complete in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Best test acc: {best_test_acc:.2f}% at epoch {best_epoch}")

    history_entry = {
        'model': args.model,
        'model_info': model_info,
        'best_epoch': best_epoch,
        'best_test_acc': best_test_acc,
        'best_test_error': 100. - best_test_acc,
    }
    history_path = os.path.join(exp_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        if isinstance(history_data, list):
            history_list = history_data
        else:
            history_list = [history_data]
    else:
        history_list = []
    history_list.append(history_entry)
    with open(history_path, 'w') as f:
        json.dump(history_list, f, indent=2)

    return 100. - best_test_acc


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 P8CNN vs Baseline Training')
    parser.add_argument('--model', type=str, default='p8cnn', choices=['p8cnn', 'baseline'],
                        help='Model: p8cnn or baseline')
    parser.add_argument('--n_channels', type=int, default=32,
                        help='Number of channels for P8CNN (default: 32)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
