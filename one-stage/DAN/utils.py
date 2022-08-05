#!/usr/bin/env Python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import json
import editdistance as ed
import os
import time
import io
import chardet
#中文编码为gbk
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class cha_encdec():
    def __init__(self, dict_file, case_sensitive=True):
        """
        :param dict_file: 记录数据集中所有可能的字符的txt文件
        :param case_sensitive: 若为True，区分大小写
        """
        self.dict = []  # 字符到数字转化的词典
        self.case_sensitive = case_sensitive
        lines = io.open(dict_file, 'r',encoding='gbk').readlines()
        #lines = io.open(dict_file).readlines()
        for lin in lines:
            line = lin.encode('utf-8') #中文时使用
            self.dict.append(line.replace('\n', ''))

    def encode(self, label_batch):
        """
        将字符标签根据self.dict转化为对应的数字形式，返回新的label， shape(nB, max_len+1)
        :param label_batch: 由一个batch的标签组成的list
        # 下面out的每一行末尾至少有一个0，这样就可通过找到0的索引而知道每一行有效标签长度
        """
        max_len = max([len(s) for s in label_batch])/3#英文时不需要/3,中文时max需要/3
        out = torch.zeros(len(label_batch), max_len + 1).long()
        for i in range(0, len(label_batch)):
            if not self.case_sensitive:
                cur_encoded = torch.tensor(
                    [self.dict.index(char.lower()) if char.lower() in self.dict else len(self.dict)
                     for char in label_batch[i]]) + 1
            else:
                #英文数据集时使用
                # cur_encoded = torch.tensor([self.dict.index(char) if char in self.dict else len(self.dict)
                #                                 for char in label_batch[i]]) + 1
                #中文数据集使用
                unilabel = label_batch[i].decode('utf-8')
                cur_encoded = [0 for k in range(len(unilabel))]
                for t in range(len(unilabel)):
                    char = unilabel[t].encode('utf-8')
                    if char in self.dict:
                        cur_encoded[t] = self.dict.index(char)
                    else:
                        cur_encoded[t] = len(self.dict)
                cur_encoded = torch.tensor(cur_encoded) + 1
            out[i][0:len(cur_encoded)] = cur_encoded
        return out


    def decode(self, net_out, length):
        # decoding prediction into text with geometric-mean probability
        # the probability is used to select the more realiable prediction when using bi-directional decoders
        """
        将DTD的预测输出解码为字符串以及返回对应的几何平均概率值
        :param net_out: DTD预测输出，shape(lenText, nclass)， lenText是batch中标签总长
        :param length: DTD预测输出的各个标签对应的长度，shape(nB)
        :return: 返回batch中各个样本的预测输出字符串，和对应的几何平均概率值
        """
        out = []
        out_prob = []
        for g in range(len(net_out)):
            net_out[g][0] = 0
        net_out = F.softmax(net_out, dim=1)  # 对第一维应用softmax，使得其输出在[0,1]之间，且之和为1，即分到每一（字符）类的概率
        torch.set_printoptions(threshold=None, edgeitems=40, linewidth=None)
        for i in range(0, length.shape[0]):
            # 以下分别为每一样本（分到最大概率的类索引，将索引转化为字符串，分到类的概率最大值，对概率最大值log求均值并指数）
            current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:, 0].tolist()
            current_text = ''.join([self.dict[_ - 1] if 0 < _ <= len(self.dict) else '' for _ in current_idx_list])
            current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]
            current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
            out.append(current_text)
            out_prob.append(current_probability)
        return out, out_prob


class Attention_AR_counter():
    def __init__(self, display_string, dict_file, case_sensitive):
        """
        :param display_string: 作用描述，string
        :param dict_file: 字符与数字转换的词典所在的txt文件路径名
        :param case_sensitive: 若为True，表示区分大小写
        """
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        self.display_string = display_string
        self.case_sensitive = case_sensitive
        self.de = cha_encdec(dict_file, case_sensitive)  # 字符数字转化dict
        self.times = 0.

    def clear(self):
        self.correct = 0
        self.total_samples = 0.
        self.distance_C = 0
        self.total_C = 0.
        self.distance_W = 0
        self.total_W = 0.
        self.times = 0.

    def add_iter(self, output, out_length, label_length, labels, s_time):
        """
        :param output: DTD的预测输出，shape(lenText, nclass)， lenText是batch中标签总长
        :param out_length: DTD预测输出的各个标签对应的长度，shape(nB)。若是训练的话，它等价于label_length
        :param label_length: 实际标签的各个长度，shape(nB)
        :param labels: 一个batch的实际的（字符串格式）标签组成的list
        """

        start = 0
        start_o = 0
        self.total_samples += label_length.size()[0]
        raw_prdts = output.topk(1)[1]  # 返回output中第一维中各个项的最大值所在的索引
        prdt_texts, prdt_prob = self.de.decode(output, out_length)

        self.times += (time.time() - s_time)
        for i in range(0, len(prdt_texts)):
            if not self.case_sensitive:
                prdt_texts[i] = prdt_texts[i].lower()
                labels[i] = labels[i].lower()
            all_words = []
            for w in labels[i].split('|') + prdt_texts[i].split('|'):
                if w not in all_words:
                    all_words.append(w)
            l_words = [all_words.index(_) for _ in labels[i].split('|')]
            p_words = [all_words.index(_) for _ in prdt_texts[i].split('|')]
            label_len = len(labels[i])
            self.distance_C += ed.eval(labels[i], prdt_texts[i][:label_len])  # 用于计算字符错误率
            self.distance_W += ed.eval(l_words, p_words)  # 用于计算词错误率
            self.total_C += len(labels[i])
            self.total_W += len(l_words)
            self.correct = self.correct + 1 if labels[i] == prdt_texts[i][:label_len] else self.correct  # 统计预测正确数

    def show(self, f=None):
        """打印显示预测准确率(相当于SER)、 AR、 CER、 WER"""
        # Accuracy for scene text.
        # CER and WER for handwritten text.
        print(self.display_string)
        if self.total_samples == 0:
            pass
        print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
            self.correct / self.total_samples,
            1 - self.distance_C / self.total_C,
            self.distance_C / self.total_C,
            self.distance_W / self.total_W))
        if f:
            f.write('{}\nAccuracy: {:.6f}\n'.format(self.display_string, self.correct / self.total_samples))
        acc = self.correct / self.total_samples  # 添加了返回准确率
        print("测试集所消耗的时间 {0:.4f}s".format(self.times/self.total_samples))
        self.clear()
        return acc


class Loss_counter():
    def __init__(self, display_interval):
        self.display_interval = display_interval
        self.total_iters = 0.
        self.loss_sum = 0

    def add_iter(self, loss):
        self.total_iters += 1
        self.loss_sum += float(loss)

    def clear(self):
        self.total_iters = 0
        self.loss_sum = 0

    def get_loss(self):
        """返回平均损失值"""
        loss = self.loss_sum / self.total_iters if self.total_iters > 0 else 0
        self.total_iters = 0
        self.loss_sum = 0
        return loss
