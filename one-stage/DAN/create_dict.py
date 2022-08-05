# coding: gbk
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np

charset_path = open("/home/abc/LAB_workspace/DAN/charset","r")
dict_path = "/home/abc/LAB_workspace/DAN/Decoupled-attention-network/dict/dic_randomChinese.txt"

charset = charset_path.read()
char_num = int(charset.split('{')[0])
chars = charset.split('{')[1]

with open(dict_path,'a') as dic:
    for i in range(char_num):
        char_o = chars.split(',',char_num-1)[i]
        char = char_o.split("'",2)[1]

        print(char)
        dic.write(char+'\n')

    dic.close()