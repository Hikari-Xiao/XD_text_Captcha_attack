# coding: utf-8
import cv2
import os
import tensorflow as tf
import numpy as np
#from chineseMap_random import chartolabel#引入字典
from chineseMap_it168 import chartolabel#引入字典

os.environ['CUDA_VISIBLE_DEVICES']='0'

typee = ['val','test','train']
type_list = ['baidu_Chinese', 'dajie',  'douban', 'renmin','it168','random']
d_t = {'baidu_Chinese':2, 'dajie':4,  'douban':4, 'renmin':2,'it168':4,'random': 5}    # 验证码的单张图文本最长长度
fill = {'baidu_Chinese':948, 'dajie':2467,  'douban':984, 'renmin':483,'it168':745,'random':3626}
types = type_list[4]


#chartolabel在chineseMap.py文件中
Image_H = 100
Image_W = 300

for tp in range(1,2):
    kind = typee[tp]
    # kind = 'val'
    print(str(types)+"---------"+str(kind))
    Seq_length = d_t[types]  # 最大长度
    Fill_label = fill[types]
    #Fill_label = 3626
    file_dir = "/home/abc/LAB_workspace/DAN/Decoupled-attention-network/data/captcha/中文/{0}/10w/{1}/".format(types,kind)
    # file_dir = "/home/abc/LAB_workspace/DAN/val-chinese/"
    files_paths = os.listdir(file_dir)
    tf_writer = tf.python_io.TFRecordWriter("tfrecords/chinese/{0}/{1}.record".format(types,kind))

    i = 0
    for p in files_paths:
        if i%100 == 0:
            print(i)
        i += 1
        if i>20000-1:
            break
        tmp_img = cv2.imread(file_dir+p)
        tmp_chrs = p.split("_")[1].split('.')[0].encode('utf-8').decode('utf-8')
        # tmp_labels = [chartolabel[c.encode('utf-8')] for c in tmp_chrs]
        tmp_labels = [chartolabel[c] for c in tmp_chrs]

        while len(tmp_labels) != Seq_length:
            tmp_labels.append(Fill_label)
        # print(str(tmp_chrs)+"----"+str(tmp_labels))

        tmp_img = cv2.resize(tmp_img,(Image_W, Image_H))
        #print tmp_chrs
        #print tmp_labels
        # cv2.imshow("1", tmp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        tmp_img = np.asarray(tmp_img, np.uint8)
        #tmp_img = tmp_img / 255.
        # print tmp_img
        tmp_img_raw = tmp_img.tobytes()
        tmp_example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_img_raw])),
            'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=tmp_labels)),
        }))
        tf_writer.write(tmp_example.SerializeToString())
    tf_writer.close()