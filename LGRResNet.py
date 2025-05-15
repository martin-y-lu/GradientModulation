import torch
import torch.nn as nn
import torch.nn.functional as F

from GModReLU import LGRLinear,LGRConv2d

class LGRBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, l=0.01, k=5.0):
        super().__init__()
        self.conv1 = LGRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, l=l, k=k)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = LGRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, l=l, k=k)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                LGRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, l=l, k=k),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class LGRBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, l=0.01, k=5.0):
        super().__init__()
        self.conv1 = LGRConv2d(in_planes, planes, kernel_size=1, bias=False, l=l, k=k)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = LGRConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, l=l, k=k)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = LGRConv2d(planes, self.expansion * planes, kernel_size=1, bias=False, l=l, k=k)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                LGRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, l=l, k=k),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class LGRResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l=0.01, k=5.0):
        super().__init__()
        self.in_planes = 64
        self.l = l
        self.k = k

        self.conv1 = LGRConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, l=l, k=k)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, l=self.l, k=self.k))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def LGRResNet18(l=0.01, k=5.0):
    return LGRResNet(LGRBasicBlock, [2, 2, 2, 2], l=l, k=k)

def LGRResNet34(l=0.01, k=5.0):
    return LGRResNet(LGRBasicBlock, [3, 4, 6, 3], l=l, k=k)

def LGRResNet50(l=0.01, k=5.0):
    return LGRResNet(LGRBottleneck, [3, 4, 6, 3], l=l, k=k)

def LGRResNet101(l=0.01, k=5.0):
    return LGRResNet(LGRBottleneck, [3, 4, 23, 3], l=l, k=k)

def LGRResNet152(l=0.01, k=5.0):
    return LGRResNet(LGRBottleneck, [3, 8, 36, 3], l=l, k=k)