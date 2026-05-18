import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, width=66):
        super().__init__()
        w = width

        self.conv1 = nn.Conv2d(3, w, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(w)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(w, w, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(w)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(w, w, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(w)
        self.relu4 = nn.ReLU(inplace=True)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def make_baseline_matched_to_p8cnn(num_classes=10, n_channels=32, search_max=128):
    from model import CIFAR_P8CNN

    ref = CIFAR_P8CNN(num_classes=num_classes, n_channels=n_channels)
    ref_params = sum(p.numel() for p in ref.parameters())
    del ref

    best_diff = 10**18
    best_w = 1

    for w in range(1, search_max + 1):
        model = BaselineCNN(num_classes=num_classes, width=w)
        params = sum(p.numel() for p in model.parameters())
        diff = abs(params - ref_params)
        if diff < best_diff:
            best_diff = diff
            best_w = w
        if diff <= 100:
            break
        del model

    model = BaselineCNN(num_classes=num_classes, width=best_w)
    baseline_params = sum(p.numel() for p in model.parameters())
    return model, best_w, ref_params, baseline_params


def count_parameters(model, verbose=True):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
    return total, trainable


if __name__ == "__main__":
    model, width, ref_params, baseline_params = make_baseline_matched_to_p8cnn()
    print(f"Baseline width: {width}")
    print(f"Reference P8CNN params: {ref_params:,}")
    print(f"Baseline params: {baseline_params:,}")
    print(f"Difference: {abs(baseline_params - ref_params):,}")
