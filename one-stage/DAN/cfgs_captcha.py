#!/usr/bin/env Python
# coding: utf-8
import torch
import torch.optim as optim
import os
from glob import glob
from dataset_scene import *
from torchvision import transforms
from DAN import *
#chinese-base CAPTCHA code with gbk

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
# reload(sys)
# sys.setdefaultencoding('gbk')
global_cfgs = {
    'state': 'Test',   # Train / Test
    'epoch': 200,
    'show_interval': 50,
    'test_interval': 5  # test_interval < Total number of iterations
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
type_list: CAPTCHA name
d_t: Maximum length of the text of captcha images
d_n: Number of character classes of the captcha
d_wh: Size of captcha
Roman-based CAPTCHA
"""
# type_list = ['360', '360_gray', 'alipay', 'apple', 'baidu', 'baidu_blue', 'baidu_red',
#              'jd', 'jd_grey', 'jd_white', 'ms', 'qqmail', 'sina', 'weibo', 'wiki']
#
# d_t = {'wiki': 10, 'weibo': 4, 'baidu_red': 4, 'apple': 5,
#        'alipay': 4, 'baidu': 4, '360': 5, 'qqmail': 4,
#        'jd_grey': 4,
#        '360_gray': 5, 'ms': 6, 'jd': 4, 'jd_white': 4,
#        'baidu_blue': 4, 'sina': 5, 'random_captcha': 10}
#
# d_n = {'wiki': 26, 'weibo': 25, 'baidu_red': 25, 'apple': 34, 'alipay': 36, 'baidu': 26, '360': 43, 'qqmail': 42,
#          'jd_grey': 22, '360_gray': 43, 'ms': 22, 'jd': 22, 'jd_white': 22, 'baidu_blue': 27, 'sina': 37,
#        'random_captcha': 62}
#
# d_wh = {'wiki': [285, 82], 'weibo': [100, 40], 'baidu_red': [100, 40], 'apple': [160, 70], 'alipay': [100, 30],
#         'baidu': [100, 40], '360': [100, 40], 'qqmail': [130, 53], 'jd_grey': [150, 36], '360_gray': [100, 40],
#         'ms': [216, 128], 'jd': [150, 36], 'jd_white': [150, 36], 'baidu_blue': [100, 40], 'sina': [100, 40],
#         'random_captcha': [200, 70]}

"""
Chinese-based CAPTCHA
"""
type_list = ['douban','dajie','baidu_Chinese','it168','renmin','random']

d_t = {'douban':4,'dajie':4,'baidu_Chinese':2,'it168':4,'renmin':2,'random':5}

d_n = {'douban':984,'dajie':2467,'baidu_Chinese':948,'it168':745,'renmin':483,'random':3626}

d_wh = {'douban':[250,40],'dajie':[80,34],'baidu_Chinese':[100,40],'it168': [150,50],'renmin':[120,32],'random':[180,50]}

types = type_list[1]  # CAPTCHA type index
print("print~~"+types)

e = "2w"
# dirs = "models/中文/{0}_fine".format(e)
dirs = "/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/re/2w"

# The following 3 lines need to be commented out during training; uncomment them during testing
models = fun(types)
i = 0  #
print("print: ", i, '  ', len(models), '  ', models[i][0])

dataset_cfgs = {
    'dataset_train': lmdbDataset,
    'dataset_train_args': {
        'roots': ['/home/abc/LAB_workspace/DAN/Decoupled-attention-network/data/captcha/{0}/{1}/train_lmdb'.format(types,e)],  # lmdb文件所在目录（若有多个数据集，就用list保存）
        'img_height': 64,
        'img_width': 128,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Train',
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
        'maxT':  d_t[types] + 2,    # maxT =  d_t[types] + 2
        # 'maxT':  5+2,             # English fine tuning label length
        'depth': 8,
        'num_channels': 64,
    },
    'DTD': DTD,
    'DTD_args': {
        'nclass': d_n[types] + 2,  # nclass = d_n[types] + 2
        # 'nclass': 3626 + 2,   # English fine tuning label length
        'nchannel': 512,
        'dropout': 0.3,
    },

    # Use these when fine-tuning the base model
    # 'init_state_dict_fe':  '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/中文/base/1_E12_I6500-12500_M0_Acc-0.98108.pth',
    # 'init_state_dict_cam': '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/中文/base/1_E12_I6500-12500_M1_Acc-0.98108.pth',
    # 'init_state_dict_dtd': '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/中文/base/1_E12_I6500-12500_M2_Acc-0.98108.pth',    # 'init_state_dict_fe':  'models/random/base/1_E3_I2600-5000_M0_Acc-0.746959797797.pth',

    # Use these when testing
    'init_state_dict_fe': models[i][0],
    'init_state_dict_cam': models[i][1],
    'init_state_dict_dtd': models[i][2],

    # Use these when training from scratch
    # 'init_state_dict_fe': None,
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
    'saving_iter_interval': 300,   # datanum / (batchsize * 2) < saving_iter_interva < datanum / batchsize
    'saving_epoch_interval': 1,

    'saving_path': '/home/abc/LAB_workspace/DAN/Decoupled-attention-network/models/re/{0}/{1}/'.format(e, types),  # model save path
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
