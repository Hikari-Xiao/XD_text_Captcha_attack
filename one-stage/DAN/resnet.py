# coding:utf-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):  # 1*1卷积，（输入通道数，输出通道数，卷积核大小，步长）
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):  # 3*3卷积
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):  # nn.Module所有神经网络模块的基类，一般torch框架下定义网络都继承该基类，并需重写forword方法
    """ResNet的基础残差块，该类定义的网络结构是：(1*1conv、BN、ReLU、3*3conv、BN)(x) + x"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, strides, compress_layer=True):
        """
        :param block: BasicBlock的实例化类（不确定）
        :param layers: 各个块中，resblock中包含的basicblock数
        :param strides: 各个块中的卷积步长
        :param compress_layer:
        """
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[1])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[3])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[4])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[5])

        self.compress_layer = compress_layer
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True))

        for m in self.modules():  # 对所有submodule进行参数初始化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))   # 使用正态分布初始化权重参数（均值，标准差）
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        :param block: 指basicBlock类实例
        :param planes: 输出通道数
        :param blocks: 包含的basicBlock个数
        :param stride: 步长
        """
        downsample = None
        # stride为2时下采样，即宽高减半。输入和输出通道不同时，改变通道数
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, multiscale=False):  # 注意，输入网络的x是四维（nB, C, H, W)，初始时，C=1
        # conv1+BN+ReLU + 3*resblock1 + 4*resblock2 + 6*resblock3 + 6*resblock4 + 3*resblock5
        out_features = []
        x = self.conv1(x)      # shape(nB, 32, H, W)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_shape = x.size()[2:]
        x = self.layer1(x)    # shape(nB, 32, H/2, W/2)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer2(x)       # shape(nB, 64, H/2, W/2)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer3(x)      # shape(nB, 128, H/4, W/4)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer4(x)       # shape(nB, 256, H/4, W/4)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer5(x)       # shape(nB, 512, H/4, W/4)
        if not self.compress_layer:
            out_features.append(x)
        else:
            if x.size()[2:] != tmp_shape:
                tmp_shape = x.size()[2:]
                out_features.append(x)
            x = self.layer6(x)
            out_features.append(x)
        return out_features   # 返回有高宽改变的几个块和最后一块的输出特征图，对于scene，即（layer1， layer3， layer5）


def resnet45(strides, compress_layer):
    model = ResNet(BasicBlock, [3, 4, 6, 6, 3], strides, compress_layer)
    return model
