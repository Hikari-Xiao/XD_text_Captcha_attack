# coding: utf-8
import cv2
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']='0'
#根据字典txt制作python文件
# file_path = "/home/abc/LAB_workspace/DAN/charset"
#
# fi = open(file_path,'r')
#
# context = fi.read()
# num = int(context.split('{')[0])
# dict = context.split('{')[1].split('}')[0]
#
# print(num)
# con = ""
#
# for i in range(num):
#     word = (dict.split(',',num-1)[i])[1:]+":"+str(i)+","
#     con += word
#     if i%5==0 and i!=0:
#         print(con)
#         con = ""

# 制作字典txt
# img_path ="/home/abc/LAB_workspace/DAN/train-chinese/"
# files_path = "/home/abc/LAB_workspace/DAN/charset"
# files = os.listdir(img_path)
# max_len = 0
# char_num = 0
# dict = []
#
# with open(files_path,mode='w',encoding='utf-8') as file_ops:
#     for file in files:
#         file_name = file.split('_')[1].split('.')[0]
#         lens = len(file_name)
#         if lens>max_len :
#             max_len = lens
#
#         for i in range(lens):
#             dict.append(file_name[i])
#
#     dictt = set(dict)
#     file_ops.write(str(len(dictt))+str(dictt))
# print(len(dictt))
# print(max_len)

file_path="/home/abc/LAB_workspace/DAN/val-chinese/"
files = os.listdir(file_path)
i=0
for file in files:
    if file.split('.')[1]=="jpg":
        old_file = file_path+file
        new_file = file_path+file.split('.')[0]+".png"
        os.rename(old_file, new_file)
    if i%100==0:
        print(i)
    i += 1
