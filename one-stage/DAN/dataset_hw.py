# coding:utf-8
from torch.utils.data import Dataset, DataLoader
import time
import math
import numpy as np
import os
import torch
import cv2
import Augment
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class LineGenerate():
    def __init__(self, IAMPath, conH, conW, augment=False, training=False):
        """
        :param IAMPath: 记录所有图片所在文件夹名的txt文件路径名。注意，Line指的是指每张图片包含一行话
        :param conH: （最终输入网络的）图高
        :param conW:  图宽
        :param augment:  若为True，表示扩充数据集
        :param training: 若为True，表示训练模式
        """
        self.training = training
        self.augment = augment
        self.conH = conH
        self.conW = conW
        standard = []
        with open(IAMPath) as f:
            for line in f.readlines():
                standard.append(line.strip('\n'))
        self.image = []   # 存储所有图片（灰度图）数据
        self.label = []   # 存储所有图片对应的文本标签
        line_prefix = '/'.join(IAMPath.split('/')[:-1]) + '/lines'  # 图片所在文件夹
        IAMLine = line_prefix + '.txt'  # 对应标签所在txt文件
        count = 0
        with open(IAMLine) as f:
            for line in f.readlines():
                elements = line.split()
                pth_ele = elements[0].split('-')
                line_tag = '%s-%s' % (pth_ele[0], pth_ele[1])
                if line_tag in standard:
                    pth = line_prefix + '/%s/%s-%s/%s.png' % (pth_ele[0], pth_ele[0], pth_ele[1], elements[0])  # 图片路径名
                    img = cv2.imread(pth, 0)  # see channel and type。以灰度图形式读取图片
                    self.image.append(img)
                    self.label.append(elements[-1])  # 存储标签（标签示例：A|MOVE|to|stop|Mr.|Gaitskell|from）
                    count += 1
        self.len = count  # 总的样本数
        self.idx = 0

    def get_len(self):
        return self.len

    def generate_line(self):
        # 若是训练，则随机抽取一样本，若测试，则从头按顺序读取样本
        if self.training:
            idx = np.random.randint(self.len)
            image = self.image[idx]
            label = self.label[idx]
        else:
            idx = self.idx
            image = self.image[idx]
            label = self.label[idx]
            self.idx += 1
        if self.idx == self.len:
            self.idx -= self.len

        # 调整图片尺寸，（宽与高保持原始比例，且高宽都不能超过预定的高宽，最终所有图片具有统一高宽conH-conW，不足部分用白色填充）
        h, w = image.shape
        imageN = np.ones((self.conH, self.conW)) * 255
        beginH = int(abs(self.conH - h) / 2)
        beginW = int(abs(self.conW - w) / 2)
        if h <= self.conH and w <= self.conW:
            imageN[beginH:beginH + h, beginW:beginW + w] = image
        elif float(h) / w > float(self.conH) / self.conW:
            newW = int(w * self.conH / float(h))
            beginW = int(abs(self.conW - newW) / 2)
            image = cv2.resize(image, (newW, self.conH))
            imageN[:, beginW:beginW + newW] = image
        elif float(h) / w <= float(self.conH) / self.conW:
            newH = int(h * self.conW / float(w))
            beginH = int(abs(self.conH - newH) / 2)
            image = cv2.resize(image, (self.conW, newH))
            imageN[beginH:beginH + newH] = image
        label = self.label[idx]

        # 对图片进行“失真或者拉伸或转换视角”以实现数据多样化
        if self.augment and self.training:
            imageN = imageN.astype('uint8')  # 转换数据类型为‘uint8’
            if torch.rand(1) < 0.3:
                imageN = Augment.GenerateDistort(imageN, random.randint(3, 8))
            if torch.rand(1) < 0.3:
                imageN = Augment.GenerateStretch(imageN, random.randint(3, 8))
            if torch.rand(1) < 0.3:
                imageN = Augment.GeneratePerspective(imageN)

        imageN = imageN.astype('float32')
        imageN = (imageN - 127.5) / 127.5  # 归一化
        return imageN, label  # 返回（图像数据，标签）


class WordGenerate():
    def __init__(self, IAMPath, conH, conW, augment=False):
        """
        :param IAMPath: 记录所有图片所在文件夹名的txt文件路径名。注意，Word指的是指每张图片包含一个单词
        :param conH: （最终输入网络的）图高
        :param conW: 图宽
        :param augment: 若为True，则扩充数据集
        """
        self.augment = augment
        self.conH = conH
        self.conW = conW
        standard = []
        with open(IAMPath) as f:
            for line in f.readlines():
                standard.append(line.strip('\n'))
        self.image = []  # 存储所有输入图像数据（灰度图）
        self.label = []  # 存储所有图像对应的单词标签

        word_prefix = '/'.join(IAMPath.split('/')[:-1]) + '/words'
        IAMWord = word_prefix + '.txt'
        count = 0
        with open(IAMWord) as f:
            for line in f.readlines():
                elements = line.split()
                pth_ele = elements[0].split('-')
                line_tag = '%s-%s' % (pth_ele[0], pth_ele[1])
                if line_tag in standard:
                    pth = word_prefix + '/%s/%s-%s/%s.png' % (pth_ele[0], pth_ele[0], pth_ele[1], elements[0])
                    img = cv2.imread(pth, 0)  # see channel and type
                    if img is not None:
                        self.image.append(img)
                        self.label.append(elements[-1])  # 标签示例：MOVE
                        count += 1
                    else:
                        print('error：the word-image is None')
                        continue;

        self.len = count  # 总的样本数

    def get_len(self):
        return self.len

    def word_generate(self):

        endW = np.random.randint(50);
        label = ''
        imageN = np.ones((self.conH, self.conW)) * 255
        imageList = []
        while True:  # 对于单词图，每次抽取一定量的图片，从左到右粘到高宽为（conH, ConW)的白图上，直到粘不下
            idx = np.random.randint(self.len)
            image = self.image[idx]
            h, w = image.shape
            beginH = int(abs(self.conH - h) / 2)
            imageList.append(image)
            if endW + w > self.conW:
                break;
            if h <= self.conH:
                imageN[beginH:beginH + h, endW:endW + w] = image
            else:
                imageN[:, endW:endW + w] = image[beginH:beginH + self.conH]

            endW += np.random.randint(60) + 20 + w
            if label == '':
                label = self.label[idx]
            else:
                label = label + '|' + self.label[idx]

        label = label

        # 对图片进行“失真或者拉伸或转换视角”以实现数据多样化
        imageN = imageN.astype('uint8')
        if self.augment:
            if torch.rand(1) < 0.3:
                imageN = Augment.GenerateDistort(imageN, random.randint(3, 8))
            if torch.rand(1) < 0.3:
                imageN = Augment.GenerateStretch(imageN, random.randint(3, 8))
            if torch.rand(1) < 0.3:
                imageN = Augment.GeneratePerspective(imageN)

        imageN = imageN.astype('float32')
        imageN = (imageN - 127.5) / 127.5
        return imageN, label


class IAMDataset(Dataset):  # 用于测试集
    def __init__(self, img_list, img_height, img_width, transform=None):
        IAMPath = img_list
        self.conH = img_height
        self.conW = img_width
        self.LG = LineGenerate(IAMPath, self.conH, self.conW)

    def __len__(self):
        return self.LG.get_len()

    def __getitem__(self, idx):
        imageN, label = self.LG.generate_line()

        imageN = imageN.reshape(1, self.conH, self.conW)
        sample = {'image': torch.from_numpy(imageN), 'label': label}

        return sample


class IAMSynthesisDataset(Dataset):  # pytorch中的数据读取，Dataset是一个抽象类，自定义的类继承该类，并需重写__getitem__方法
    def __init__(self, img_list, img_height, img_width, augment=False, transform=None):
        """
        :param img_list: txt文件，记录了所有图像所在文件夹名
        :param img_height: 图像的高度
        :param img_width: 图像的宽度
        :param augment: 若为True，表示对图片进行改变以扩充数据集
        :param transform:
        """
        self.training = True
        self.augment = augment
        IAMPath = img_list
        self.conH = img_height
        self.conW = img_width
        self.LG = LineGenerate(IAMPath, self.conH, self.conW, self.augment, self.training)
        self.WG = WordGenerate(IAMPath, self.conH, self.conW, self.augment)

    def __len__(self):  # 返回样本量
        return self.WG.get_len()

    def __getitem__(self, idx):   # 该方法是定义样本读取方式，每执行一次，读取一个样本
        if np.random.rand() < 0.5:
            imageN, label = self.LG.generate_line()
        else:
            imageN, label = self.WG.word_generate()

        imageN = imageN.reshape(1, self.conH, self.conW)  # 将图像从转化为3维（1×H×W）
        sample = {'image': torch.from_numpy(imageN), 'label': label}  # 将图像数据转化为tensor
        return sample  # 返回一个样本，格式为{'image': 图, 'label': 标签}
