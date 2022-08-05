#!/usr/bin/env Python
# coding=utf-8
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import datetime
import time
import json
from PIL import Image
import os

# ------------------------
# 指定可见的GPU号(注：虽然设定的可见GPU号为‘1’，对应在程序的编号还是从0开始的；此外，最好将其写在代码顶端，如需导入其他同目录下的模块)
# 则也需在这些导入模块之前，这样可以不用在导入模块文件中写改行。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import *
# import cfgs_hw as cfgs
# import cfgs_scene as cfgs
import cfgs_captcha as cfgs


# ------------------------
def display_cfgs(models):
    print('global_cfgs')
    cfgs.showcfgs(cfgs.global_cfgs)
    print('dataset_cfgs')
    cfgs.showcfgs(cfgs.dataset_cfgs)
    print('net_cfgs')
    cfgs.showcfgs(cfgs.net_cfgs)
    print('optimizer_cfgs')
    cfgs.showcfgs(cfgs.optimizer_cfgs)
    print('saving_cfgs')
    cfgs.showcfgs(cfgs.saving_cfgs)
    for model in models:
        print(model)


def flatten_label(target):
    """将2维标签（batch）拉伸为1维，每一标签最后一个值都是0， 并返回各个标签的有效长度组成的tensor"""
    label_flatten = []
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0) + 1]
        label_length.append(cur_label.index(0) + 1)
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    return (label_flatten, label_length)


def Train_or_Eval(models, state='Train'):
    for model in models:
        if state == 'Train':
            model.train()  # 只影响dropout和batchNorm
        else:
            model.eval()


def Zero_Grad(models):
    """梯度清零"""
    for model in models:
        model.zero_grad()


def Updata_Parameters(optimizers, frozen):
    """利用计算的梯度更新参数。frozen用于如果有某个模型准备不更新（即frozen），就不执行该模型的参数更新"""
    for i in range(0, len(optimizers)):
        if i not in frozen:
            optimizers[i].step()



# ---------------------dataset
def load_dataset():
    train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])  # 实例化自定义的Dataset
    train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train'])  # 数据集加载的迭代器

    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])
    # pdb.set_trace()
    return (train_loader, test_loader)


# ---------------------network
def load_network():
    model_fe = cfgs.net_cfgs['FE'](**cfgs.net_cfgs['FE_args'])  # 特征提取

    cfgs.net_cfgs['CAM_args']['scales'] = model_fe.Iwantshapes()  # 特征提取这一步返回的各个块的输出特征图尺寸
    model_cam = cfgs.net_cfgs['CAM'](**cfgs.net_cfgs['CAM_args'])  # 卷积对齐模块

    model_dtd = cfgs.net_cfgs['DTD'](**cfgs.net_cfgs['DTD_args'])  # 解耦文本解码器

    # 如果这三个模块中某一模块并非从头开始训练（即已有训练好的参数），则加载相应的模型
    if cfgs.net_cfgs['init_state_dict_fe'] != None:
        model_fe.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_fe'], map_location='cuda:0'))
    if cfgs.net_cfgs['init_state_dict_cam'] != None:
        model_cam.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_cam'], map_location='cuda:0'))
    if cfgs.net_cfgs['init_state_dict_dtd'] != None:
        model_dtd.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_dtd'], map_location='cuda:0'))

    model_fe.cuda()  # 将内存中的数据迁到GPU，用GPU进行计算
    model_cam.cuda()
    model_dtd.cuda()
    return (model_fe, model_cam, model_dtd)


# ----------------------optimizer
def generate_optimizer(models):
    out = []
    scheduler = []
    for i in range(0, len(models)):
        out.append(cfgs.optimizer_cfgs['optimizer_{}'.format(i)](
            models[i].parameters(),
            **cfgs.optimizer_cfgs['optimizer_{}_args'.format(i)]))
        scheduler.append(cfgs.optimizer_cfgs['optimizer_{}_scheduler'.format(i)](
            out[i],
            **cfgs.optimizer_cfgs['optimizer_{}_scheduler_args'.format(i)]))
    return tuple(out), tuple(scheduler)


# ---------------------testing stage
def test(test_loader, model, tools, f=None):
    """将测试集输入网络，并计算打印预测准确率(相当于SER)、 AR、 CER、 WER"""
    Train_or_Eval(model, 'Eval')
    for sample_batched in test_loader:
        s_time = time.time()
        data = sample_batched['image']  # shape(nB, C, H, W)
        label = sample_batched['label']
        target = tools[0].encode(label)  # 将标签由字符串转化为数字序列，shape(nB, max_len+1)
        data = data.cuda()
        target = target
        label_flatten, length = tools[1](target)
        target, label_flatten = target.cuda(), label_flatten.cuda()

        features = model[0](data)
        A = model[1](features)
        output, out_length = model[2](features[-1], A, target, length, True)
        tools[2].add_iter(output, out_length, length, label, s_time)
    acc = tools[2].show(f)
    Train_or_Eval(model, 'Train')
    return acc


# ---------------------------------------------------------
# --------------------------Begin--------------------------
# ---------------------------------------------------------
if __name__ == '__main__':

    # prepare nets, optimizers and data
    model = load_network()
    # display_cfgs(model)  # 打印出模型的配置
    optimizers, optimizer_schedulers = generate_optimizer(model)  # 返回三个模块对应的优化器和（学习率调度器）
    criterion_CE = nn.CrossEntropyLoss().cuda()  # 交叉熵损失
    train_loader, test_loader = load_dataset()  # 训练集和测试集的数据加载器
    print('preparing done')
    # --------------------------------
    # prepare tools
    train_acc_counter = Attention_AR_counter('train accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                             cfgs.dataset_cfgs['case_sensitive'])
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])
    loss_counter = Loss_counter(cfgs.global_cfgs['show_interval'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    # ---------------------------------
    if cfgs.global_cfgs['state'] == 'Test':
        print("starting test!!!!!!!!!!!!!!!")
        test((test_loader),
             model,
             [encdec,
              flatten_label,
              test_acc_counter])
        exit()
    # --------------------------------

    fp = open(cfgs.saving_cfgs['saving_acc_loss'], 'a+')
    fp.truncate()  # 清空文件

    """
    一般整个迭代循环过程：
    1）将数据集输入网络进行前向传播；
    2）梯度清零（这一步需要在梯度更新之前）；
    3）计算预测输出与真实标签之间的loss（一般也会计算预测效果）；
    4）计算梯度；
    5）利用梯度和学习率等参数更新模型参数值。
    直到满足要求，停止迭代。
    """
    new_acc = -0.1
    total_iters = len(train_loader)
    print("print~~~ total_iter: ", total_iters)
    start = time.time()
    for nEpoch in range(0, cfgs.global_cfgs['epoch']):
        for batch_idx, sample_batched in enumerate(train_loader):
            s_time = time.time()
            # data prepare
            data = sample_batched['image']  # 图片数据，shape(nB, C, H, W), C=1
            label = sample_batched['label']  # 标签，shape(nB)
            target = encdec.encode(label)  # 将字符串标签转化为数字形式，shape(nB, maxlen+1),maxlen表示所有标签的最长长度
            Train_or_Eval(model, 'Train')  # 训练模式
            data = data.cuda()  # 使用GPU计算
            label_flatten, length = flatten_label(target)  # 将一batch标签从2维拉伸为1维，并用length存储每一标签的长度
            target, label_flatten = target.cuda(), label_flatten.cuda()

            # net forward。将数据输入网络模型（特征编码、CAM、DTD），返回预测输出shape(lenText, nclass)和对应的注意力图
            features = model[0](data)
            A = model[1](features)
            output, attention_maps = model[2](features[-1], A, target, length)
            # computing accuracy and loss。计算准确率和损失值
            train_acc_counter.add_iter(output, length.long(), length, label, s_time)
            loss = criterion_CE(output, label_flatten)  # label_flatten的shape(lenText)
            loss_counter.add_iter(loss)

            # update network。反向传播，更新网络参数
            Zero_Grad(model)  # 梯度清零
            loss.backward()  # 反向传播计算梯度
            # 梯度裁剪。先计算参数的总范数，然后计算设定的最大范数max_norm与总范数比值，若是小于1，就将所有参数梯度值乘以该比值。
            nn.utils.clip_grad_norm_(parameters=model[0].parameters(), max_norm=20, norm_type=2)
            nn.utils.clip_grad_norm_(model[1].parameters(), 20, 2)
            nn.utils.clip_grad_norm_(model[2].parameters(), 20, 2)
            Updata_Parameters(optimizers, frozen=[])  # 利用计算的梯度更新参数

            # visualization and saving。
            # 每隔50迭代次数，打印一次训练集的损失值和测试结果
            if batch_idx % cfgs.global_cfgs['show_interval'] == 0 and batch_idx != 0:
                print(datetime.datetime.now().strftime('%H:%M:%S'))  # 获取当前时间点
                losses = loss_counter.get_loss()
                print('Epoch: {}, Iter: {}/{}, Loss dan: {}'.format(
                    nEpoch,
                    batch_idx,
                    total_iters,
                    losses))

                fp.write("time：{}\n".format(datetime.datetime.now().strftime('%H:%M:%S')))
                fp.write("Epoch: {}, Iter: {}/{}, Loss dan: {}\n".format(nEpoch, batch_idx, total_iters, losses))
                train_acc_counter.show(fp)

            # 每隔***次迭代，打印一次测试集上的效果
            if nEpoch % cfgs.saving_cfgs['saving_epoch_interval'] == 0 and \
                    batch_idx % cfgs.saving_cfgs['saving_iter_interval'] == 0 and \
                    batch_idx != 0:
                acc = test((test_loader), model, [encdec, flatten_label, test_acc_counter], fp)
                if acc > new_acc:
                    new_acc = acc
                    for i in range(0, len(model)):
                        torch.save(model[i].state_dict(),
                                   cfgs.saving_cfgs['saving_path'] + 'E{}_I{}-{}_M{}_Acc-{}.pth'.format(
                                       nEpoch, batch_idx, total_iters, i, new_acc))
                        print("saving model ok!!!!!!!!!!!!!!!!!")
                # elif nEpoch > 10 and nEpoch % 10 == 0:
                #     for i in range(0, len(model)):
                #         torch.save(model[i].state_dict(),
                #                    cfgs.saving_cfgs['saving_path'] + 'E{}_I{}-{}_M{}_Acc-{}.pth'.format(
                #                        nEpoch, batch_idx, total_iters, i, acc))

        Updata_Parameters(optimizer_schedulers, frozen=[])  # 当epoch每达到一个milestone时，用gamma乘以lr实现学习率衰减。
    end = time.time()
    print("total time: ", end - start)
    fp.write("total time: {0}\n".format(end - start))
    fp.close()
