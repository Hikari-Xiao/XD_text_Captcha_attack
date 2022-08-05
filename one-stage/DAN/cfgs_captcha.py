#!/usr/bin/env Python
# coding: utf-8
import torch
import torch.optim as optim
import os
from glob import glob
from dataset_scene import *
from torchvision import transforms
from DAN import *
#中文编码为gbk

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
# reload(sys)
# sys.setdefaultencoding('gbk')
global_cfgs = {
    'state': 'Test',   # 训练时是Train，测试时是Test
    'epoch': 200,
    'show_interval': 50,
    'test_interval': 5  # 设置一个小于总的迭代次数的值
}


def fun(t):
    paths = glob(os.path.join(dirs, t, '*'))
    models = {}
    for pp in paths:
        p = os.path.basename(pp)
        print(p)
        a, b, c = p.split('-')
        if a+c in models.keys():
            models[a+c].append(pp)
        else:
            models[a+c] = [pp]
    for k in models.keys():
        models[k].sort(key=lambda x: int(x.split('_')[-2][1]))
    m = models.values()
    m.sort(key=lambda x: float(x[0].split('-')[-1][:-4]), reverse=True)
    return m

"""
根据这个词典，训练哪一个数据集的验证码，就对应修改dataset_train_args.roots，dataset_test_args.roots，dict_dir，nclass（是下面的
 数字+2），'init_state_dict_fe'， 'init_state_dict_cam'， 'init_state_dict_dtd'。
"""
# type_list = ['360', '360_gray', 'alipay', 'apple', 'baidu', 'baidu_blue', 'baidu_red',
#              'jd', 'jd_grey', 'jd_white', 'ms', 'qqmail', 'sina', 'weibo', 'wiki']
#
# d_t = {'wiki': 10, 'weibo': 4, 'baidu_red': 4, 'apple': 5,
#        'alipay': 4, 'baidu': 4, '360': 5, 'qqmail': 4,
#        'jd_grey': 4,
#        '360_gray': 5, 'ms': 6, 'jd': 4, 'jd_white': 4,
#        'baidu_blue': 4, 'sina': 5, 'random_captcha': 10}    # 验证码的单张图文本最长长度
#
# d_n = {'wiki': 26, 'weibo': 25, 'baidu_red': 25, 'apple': 34, 'alipay': 36, 'baidu': 26, '360': 43, 'qqmail': 42,
#          'jd_grey': 22, '360_gray': 43, 'ms': 22, 'jd': 22, 'jd_white': 22, 'baidu_blue': 27, 'sina': 37,
#        'random_captcha': 62}    # 验证码的字符类别数
#
# d_wh = {'wiki': [285, 82], 'weibo': [100, 40], 'baidu_red': [100, 40], 'apple': [160, 70], 'alipay': [100, 30],
#         'baidu': [100, 40], '360': [100, 40], 'qqmail': [130, 53], 'jd_grey': [150, 36], '360_gray': [100, 40],
#         'ms': [216, 128], 'jd': [150, 36], 'jd_white': [150, 36], 'baidu_blue': [100, 40], 'sina': [100, 40],
#         'random_captcha': [200, 70]}   # 验证码的尺寸

type_list = ['douban','dajie','baidu_Chinese','it168','renmin','random']

d_t = {'douban':4,'dajie':4,'baidu_Chinese':2,'it168':4,'renmin':2,'random':5}    # 验证码的单张图文本最长长度

d_n = {'douban':984,'dajie':2467,'baidu_Chinese':948,'it168':745,'renmin':483,'random':3626}    # 验证码的字符类别数

d_wh = {'douban':[250,40],'dajie':[80,34],'baidu_Chinese':[100,40],'it168': [150,50],'renmin':[120,32],'random':[180,50]}   # 验证码的尺寸

types = type_list[1]  # 要训练或者测试哪一种验证码，就在type_list中找到对应的索引，写到这里
print("print~~"+types)

e = "2w"  # e根据需要需求可以是500、1000、8500、500_0、1000_0、random
# dirs = "models/中文/{0}_fine".format(e)
dirs = "/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/re/2w"

# 在训练时需要注释掉下面3行；测试时，取消注释，因为训练过程中可能会保存多余1个的模型，所以对它们的结果都得测试一次，哪个最高就保留哪
# 个，下面的i取值是0到len(models)-1。
models = fun(types)
i = 0  #
print("print: ", i, '  ', len(models), '  ', models[i][0])

dataset_cfgs = {
    'dataset_train': lmdbDataset,
    'dataset_train_args': {
        'roots': ['/home/abc/LAB_workspace/DAN/Decoupled-attention-network/data/captcha/{0}/{1}/train_lmdb'.format(types,e)],  # lmdb文件所在目录（若有多个数据集，就用list保存）
        'img_height': 64,   # 图高和图宽最好设置为2的n次方
        'img_width': 128,
        'transform': transforms.Compose([transforms.ToTensor()]),  # transforms.Compose()里面是一组对图像变换的操作
        'global_state': 'Train',  # transforms.ToTensor()将PIL图像转化为shape(C, H, W)，值范围[0.0, 1.0]的tensor。
    },
    'dataloader_train': {
        'batch_size': 40,
        'shuffle': True,
        'num_workers': 3,
    },

    'dataset_test': lmdbDataset,
    'dataset_test_args': {
        'roots': ['/home/abc/LAB_workspace/DAN/Decoupled-attention-network/data/captcha/{0}/{1}/test_lmdb'.format(types,e)],  # lmdb文件所在目录
        'img_height':64,
        'img_width': 128,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },
    'dataloader_test': {
        'batch_size': 40,
        'shuffle': False,
        'num_workers': 3,
    },

    'case_sensitive': True,
    'dict_dir': '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/dict/dic_{0}.txt'.format(types)    # 'dict/dic_{0}.txt'.format(types)
}

net_cfgs = {
    'FE': Feature_Extractor,
    'FE_args': {
        'strides': [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
        # 'strides': [(1, 1), (2, 2), (2, 2), (2, 1), (2, 1), (2, 1)],
        'compress_layer': False,
        'input_shape': [1, 64, 128],  # C x H x W
    },
    'CAM': CAM,
    'CAM_args': {
        'maxT':  d_t[types] + 2,    # 设置一个大于数据集所有样本上的标签的最长值  d_t[types] + 2
        # 'maxT':  5+2,             # 英语微调标签长度
        'depth': 8,
        'num_channels': 64,
    },
    'DTD': DTD,
    'DTD_args': {
        'nclass': d_n[types] + 2,  # 类别数，字符集合总数加上未知和结束符  d_n[types] + 2
        # 'nclass': 3626 + 2,   # 英语微调类别数
        'nchannel': 512,
        'dropout': 0.3,
    },

    # 在之前的模型的基础上微调时，用这个，表示在如下表示的模型基础上微调
    # 'init_state_dict_fe':  '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/中文/base/1_E12_I6500-12500_M0_Acc-0.98108.pth',
    # 'init_state_dict_cam': '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/中文/base/1_E12_I6500-12500_M1_Acc-0.98108.pth',
    # 'init_state_dict_dtd': '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/中文/base/1_E12_I6500-12500_M2_Acc-0.98108.pth',    # 'init_state_dict_fe':  'models/random/base/1_E3_I2600-5000_M0_Acc-0.746959797797.pth',

    # 测试时，用这个
    'init_state_dict_fe': models[i][0],
    'init_state_dict_cam': models[i][1],
    'init_state_dict_dtd': models[i][2],

    # 没有预训练模型，从开始训练的话，用这个
    # 'init_state_dict_fe': None,  # 这三个分别对应于三个模块的模型的保存路径
    # 'init_state_dict_cam': None,
    # 'init_state_dict_dtd': None,
}

optimizer_cfgs = {
    # optim for FE
    'optimizer_0': optim.Adadelta,
    'optimizer_0_args': {
        'lr': 1.0,
    },

    'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_0_scheduler_args': {
        'milestones': [350, 200],
        'gamma': 0.4,
    },

    # optim for CAM
    'optimizer_1': optim.Adadelta,
    'optimizer_1_args': {
        'lr': 1.0,
    },
    'optimizer_1_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_1_scheduler_args': {

        'milestones': [350, 200],
        'gamma': 0.4,
    },

    # optim for DTD
    'optimizer_2': optim.Adadelta,
    'optimizer_2_args': {
        'lr': 1.0,
    },
    'optimizer_2_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_2_scheduler_args': {
        'milestones': [350, 200],
        'gamma': 0.4,
    },
}

saving_cfgs = {
    'saving_iter_interval': 300,   # 该值介于 单个epoch内的迭代次数的一半 到 迭代次数（等于样本数/batchsize）
    'saving_epoch_interval': 1,

    'saving_path': '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/re/{0}/{1}/'.format(e, types),  # 保存模型路径
    'saving_acc_loss': '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/re/{0}/{1}_acc_loss256.txt'.format(e, types)
}


def mkdir(path_):
    paths = path_.split('/')
    command_str = 'mkdir '
    for i in range(0, len(paths) - 1):
        command_str = command_str + paths[i] + '/'
    command_str = command_str[0:-1]
    os.system(command_str)


def showcfgs(s):
    for key in s.keys():
        print(key, s[key])
    print('')


mkdir(saving_cfgs['saving_path'])
