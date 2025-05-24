import torch
import torch.nn as nn
import torch.nn.functional as F

from GEDReLU import GEDReLU

class GEDBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, l=0.01, k1=1.0, k2=1.0, p=1.0, act_class = GEDReLU):
        super().__init__()
        self.is_GED = True
        self.act1 = act_class(l=l, k1=k1, k2=k2, p=p)
        self.act2 = act_class(l=l, k1=k1, k2=k2, p=p)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
            
    def update_s(self,beta = 0.99):
        for module in self.children():
            if hasattr(module,"is_GED"):
                module.update_s(beta)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class GEDBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, l=0.01, k1=1.0, k2=1.0, p=1.0, act_class = GEDReLU):
        super().__init__()
        self.is_GED = True
        self.act1 = act_class(l=l, k1=k1, k2=k2, p=p)
        self.act2 = act_class(l=l, k1=k1, k2=k2, p=p)
        self.act3 = act_class(l=l, k1=k1, k2=k2, p=p)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out

class GEDSequential(nn.Sequential):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.is_GED = True
    
    def update_s(self,beta = 0.99):
        for module in self.children():
            if hasattr(module,"is_GED"):
                module.update_s(beta)
    

class GEDResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l=0.01, k1=1.0, k2 = 1.0, p=1.0, act_class = GEDReLU):
        super().__init__()
        self.is_GED = True
        self.in_planes = 64
        self.l = l
        self.k1 = k1
        self.k2 = k2
        self.p = p

        self.act = act_class(l=l, k1=k1, k2=k2, p=p)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act_class = act_class)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act_class = act_class)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act_class = act_class)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, act_class = act_class)

        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
    def update_s(self,beta = 0.99):
        for module in self.children():
            if hasattr(module,"is_GED"):
                module.update_s(beta)

    def _make_layer(self, block, planes, num_blocks, stride, act_class):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, l=self.l, k1=self.k1, k2 = self.k2, p=self.p, act_class = act_class))
            self.in_planes = planes * block.expansion
        return GEDSequential(*layers)
        

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def initialize_ged_resnet(module):
    if isinstance(module, nn.Conv2d):
        # print(f"Initializing Conv: {module}")
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        # print(f"Initializing Linear: {module}")
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        # print(f"Initializing BN: {module}")
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def GEDResNet18(l=0.01, k1=1.0, k2=1.0, p=1.0, act_class = GEDReLU):
    return GEDResNet(GEDBasicBlock, [2, 2, 2, 2], l=l, k1=k1, k2 = k2, p=p, act_class = act_class)

def GEDResNet34(l=0.01, k1=1.0, k2=1.0, p=1.0, act_class = GEDReLU):
    return GEDResNet(GEDBasicBlock, [3, 4, 6, 3], l=l,k1=k1, k2 = k2, p=p, act_class = act_class)

def GEDResNet50(l=0.01, k1=1.0, k2=1.0, p=1.0, act_class = GEDReLU):
    return GEDResNet(GEDBottleneck, [3, 4, 6, 3], l=l, k1=k1, k2 = k2, p=p, act_class = act_class)

def GEDResNet101(l=0.01, k1=1.0, k2=1.0, p=1.0, act_class = GEDReLU):
    return GEDResNet(GEDBottleneck, [3, 4, 23, 3], l=l, k1=k1, k2 = k2, p=p, act_class = act_class)

def GEDResNet152(l=0.01, k1=1.0, k2=1.0, p=1.0, act_class = GEDReLU):
    return GEDResNet(GEDBottleneck, [3, 8, 36, 3], l=l, k1=k1, k2 = k2, p=p, act_class = act_class)