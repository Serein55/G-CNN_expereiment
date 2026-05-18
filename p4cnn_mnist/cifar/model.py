import torch
import torch.nn as nn
import escnn.nn as enn
from escnn import gspaces


class CIFAR_P8CNN(enn.EquivariantModule):
    def __init__(self, num_classes=10, n_channels=32):
        super(CIFAR_P8CNN, self).__init__()

        self.gspace = gspaces.rot2dOnR2(N=8)

        self.in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        self.reg_type = enn.FieldType(self.gspace, [self.gspace.regular_repr] * n_channels)

        self.conv1 = enn.R2Conv(self.in_type, self.reg_type, kernel_size=5, padding=2, bias=False)
        self.bn1 = enn.InnerBatchNorm(self.reg_type)
        self.relu1 = enn.ReLU(self.reg_type, inplace=True)

        self.conv2 = enn.R2Conv(self.reg_type, self.reg_type, kernel_size=5, padding=2, bias=False)
        self.bn2 = enn.InnerBatchNorm(self.reg_type)
        self.relu2 = enn.ReLU(self.reg_type, inplace=True)

        self.pool1 = enn.PointwiseMaxPool2D(self.reg_type, kernel_size=2, stride=2)

        self.conv3 = enn.R2Conv(self.reg_type, self.reg_type, kernel_size=3, padding=1, bias=False)
        self.bn3 = enn.InnerBatchNorm(self.reg_type)
        self.relu3 = enn.ReLU(self.reg_type, inplace=True)

        self.conv4 = enn.R2Conv(self.reg_type, self.reg_type, kernel_size=3, padding=1, bias=False)
        self.bn4 = enn.InnerBatchNorm(self.reg_type)
        self.relu4 = enn.ReLU(self.reg_type, inplace=True)

        self.adaptive_pool = enn.PointwiseAdaptiveAvgPool2D(self.reg_type, output_size=(1, 1))
        self.gpool = enn.GroupPooling(self.reg_type)

        self.fc = nn.Linear(n_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.in_type)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))

        x = self.adaptive_pool(x)
        x = self.gpool(x)

        x = x.tensor.flatten(1)
        return self.fc(x)

    def evaluate_output_shape(self, input_shape):
        return (input_shape[0], self.fc.out_features)


def count_parameters(model, verbose=True):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
    return total, trainable, total
