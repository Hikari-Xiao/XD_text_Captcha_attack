# coding: utf-8
import cv2
import os
import tensorflow as tf
import numpy as np

chartolabel = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,
               'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,
               'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,
               't':19,'u':20,'v':21,'w':22,'x':23,'y':24,
               'z':25,'A':26,'B':27,'C':28,'D':29,'E':30,
               'F':31,'G':32,'H':33,'I':34,'J':35,'K':36,
               'L':37,'M':38,'N':39,'O':40,'P':41,'Q':42,
               'R':43,'S':44,'T':45,'U':46,'V':47,'W':48,
               'X':49,'Y':50,'Z':51,'0':52,'1':53,'2':54,
               '3':55,'4':56,'5':57,'6':58,'7':59,'8':60,
               '9':61,'#':62}

type_list = ['360', '360_gray',  'apple', 'baidu','baidu_blue','jd', 'qqmail','sina','weibo']

d_t = {'wiki': 10, 'weibo': 4, 'baidu_red': 4, 'apple': 5,
       'alipay': 4, 'baidu': 4, '360': 5, 'qqmail': 4,
       'jd_grey': 4,'360_gray': 5, 'ms': 6, 'jd': 4, 'jd_white': 4,
       'baidu_blue': 4, 'sina': 5, 'random_captcha': 10}    # The maximum length of the text of a CAPTCHA image
Image_H = 150
Image_W = 500
Fill_label = 62  # class num of characters

for tp in range(0,9):
    types = type_list[tp]
    print(types)
    Seq_length = d_t[types]
    file_dir = "/home/abc/xcx/captcha/{0}/train/".format(types)
    files_paths = os.listdir(file_dir)
    tf_writer = tf.python_io.TFRecordWriter("/home/abc/LAB_workspace/attention_cap/tfrecords/{0}/8500_train.record".format(types))
    i = 0
    for p in files_paths:
        if i%10 == 0:
            print(i)
        i += 1
        if i>8500-1:
            break
        tmp_img = cv2.imread(file_dir+p)
        # tmp_chrs = p.split("_")[1].split("_")[0]
        tmp_chrs = p.split("_")[1].split(".")[0]
        tmp_labels = [chartolabel[c] for c in tmp_chrs]
        while len(tmp_labels) != Seq_length:
            tmp_labels.append(Fill_label)
        #
        tmp_img = cv2.resize(tmp_img,(Image_W, Image_H))
        #print tmp_img.shape
        #print tmp_chrs
        #print tmp_labels
        #cv2.imshow("1", tmp_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
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
