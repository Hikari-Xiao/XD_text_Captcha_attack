# coding:utf-8
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import pdb
import os
import cv2
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#中文编码为gbk
import sys
reload(sys)
sys.setdefaultencoding('gbk')

class lmdbDataset(Dataset):
    """pytorch中的数据读取，Dataset是一个抽象类，自定义的类继承该类，并需重写__getitem__方法"""
    def __init__(self, roots=None, ratio=None, img_height=32, img_width=128,
                 transform=None, global_state='Test'):
        """
        :param roots: 数据集生成的lmdb文件路径，因为不止一个数据集，所以用list表示
        :param ratio:
        :param img_height: 图高
        :param img_width: 图宽
        :param transform: 对（PIL）图像进行变换操作
        :param global_state: Train或Test
        """
        self.envs = []
        self.nSamples = 0  # 统计所有数据集的总样本数
        self.lengths = []  # 存储各个数据集中的总样本数
        self.ratio = []
        self.global_state = global_state
        for i in range(0, len(roots)):  # 读取lmdb中的数据
            env = lmdb.open(
                path=roots[i],    # lmdb存储路径
                max_readers=1,    # 同时读的最大事务数。
                readonly=True,    # 如果为True，则禁止任何写操作
                lock=False,       # 如果为False，则不要执行任何locking操作。
                readahead=False,  # 如果为False, LMDB将禁用OS文件系统的预读机制
                meminit=False)    # 如果为False, LMDB将不会在将缓冲区写入磁盘之前对它们进行零初始化。
            if not env:
                print('cannot creat lmdb from %s' % (root))
                sys.exit(0)
            with env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                self.nSamples += nSamples
            self.lengths.append(nSamples)
            self.envs.append(env)

        if ratio != None:
            assert len(roots) == len(ratio), 'length of ratio must equal to length of roots!'
            for i in range(0, len(roots)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:  # 记录各个数据集样本量与总样本量的比例
            for i in range(0, len(roots)):
                self.ratio.append(self.lengths[i] / float(self.nSamples))

        self.transform = transform
        self.maxlen = max(self.lengths)
        self.img_height = img_height
        self.img_width = img_width
        self.target_ratio = img_width / float(img_height)

    def __fromwhich__(self):
        rd = random.random()
        total = 0
        for i in range(0, len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i

    def keepratio_resize(self, img):
        """图像尺寸调整，最终图像宽高均为self.img_width，self.img_height，不足部分用白色填充"""
        cur_ratio = img.size[0] / float(img.size[1])  # 宽/高，img是PIL格式

        mask_height = self.img_height
        mask_width = self.img_width
        img = np.array(img)
        if len(img.shape) == 3:  # 如果图像是rgb格式的，就转化为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if cur_ratio > self.target_ratio:  # 如果图像原宽高比大于预定宽高比，
            cur_target_height = self.img_height
            cur_target_width = self.img_width
        else:
            cur_target_height = self.img_height
            cur_target_width = int(self.img_height * cur_ratio)
        img = cv2.resize(img, (cur_target_width, cur_target_height))  # 调整图片尺寸
        ###########################
        # img = cv2.resize(img, (182, self.img_height))  # 调整图片尺寸
        ##########################
        start_x = int((mask_height - img.shape[0]) / 2)
        start_y = int((mask_width - img.shape[1]) / 2)
        mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
        img = mask
        return img

    def __len__(self):
        """返回总样本数"""
        return self.nSamples

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0, self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:  # 读取

            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(key=img_key.encode())  # 先将key由字符串编码为字节格式，然后根据key获得相应的value，即图像数据
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)  # 打开图片，PIL格式，二维
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))  # 获取标签，str
            if len(label) > 17 and self.global_state == 'Train':  # 标签长度要小于maxT
                print('sample too long')
                return self[index + 1]
            try:
                img = self.keepratio_resize(img)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            img = img[:, :, np.newaxis]  # 给img增加一个维度
            if self.transform:  # 进行图像变换操作，并将其转化为tensor
                img = self.transform(img)
            sample = {'image': img, 'label': label}
            return sample  # 每执行一次，返回一个样本。格式：{'image': img, 'label': label}
