import os
import tensorflow as tf
from PIL import Image
import numpy as np
file_dir = "test_imgs/"
txt_dir = "test.txt"
IMG_WIDTH = 100
IMG_HEIGHT = 40
tf_writer = tf.python_io.TFRecordWriter("tfrecords/test.records")
f = open(txt_dir,"r")
tmp_str = f.readline()
i = 0
while tmp_str!="":#763200
    print i
    i+=1
    p = tmp_str.split(" ")[0]
    first_label = tmp_str.split(" ")[1]
    first_label = int(first_label)
    second_label = tmp_str.split(" ")[2][:-1]
    second_label = int(second_label)
    print first_label,second_label
    print "train_imgs/" + p
    tmp_img = Image.open("test_imgs/"+p)
    tmp_img = tmp_img.resize((IMG_WIDTH,IMG_HEIGHT))
    tmp_img = np.asarray(tmp_img,np.uint8)
    tmp_img_raw = tmp_img.tobytes()

    # bytes_list = tf.train.BytesList(tmp_img_raw)
    tmp_example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_img_raw])),
        'two_labels':tf.train.Feature(int64_list=tf.train.Int64List(value=[first_label,second_label])),
    }))
    tf_writer.write(tmp_example.SerializeToString())
    tmp_str = f.readline()
tf_writer.close()
