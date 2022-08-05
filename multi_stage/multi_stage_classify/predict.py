#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import cv2
import os
import glob
import slim.nets.inception_v3 as inception_v3
import slim.nets.vgg as vgg
import slim.nets.resnet_v1 as resnet_v1
import slim.nets.lenet as lenet
import slim.nets.alexnet as alexnet

from create_tf_record import *
import tensorflow.contrib.slim as slim
from map.english_map import label_to_char, chartolabel


def  predict(models_path,image_dir,labels_nums, data_format, model, answer_txt):
    [batch_size, resize_height, resize_width, depths] = data_format

    # labels = np.loadtxt(labels_filename, str, delimiter='\t')
    # print(labels)
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    if model == 'inception_v3':
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)
    elif model == 'vgg':
        with slim.arg_scope(vgg.vgg_arg_scope()):
            out, end_points = vgg.vgg_16(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0, is_training=False)
    elif model == "resnet":
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            out, end_points = resnet_v1.resnet_v1_50(inputs=input_images, num_classes=labels_nums,
                                                      is_training=False, global_pool=True)
    elif model == "lenet":
        with slim.arg_scope(lenet.lenet_arg_scope()):
            out, end_points = lenet.lenet(images=input_images, num_classes=labels_nums, dropout_keep_prob=1.0,
                                          is_training=False)
    elif model == "alexnet":
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            out, end_points = alexnet.alexnet_v2(inputs=input_images, num_classes=labels_nums, is_training= False,
                                                 dropout_keep_prob=1.0)

    score = tf.nn.softmax(out,name='pre')
    class_id = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)
    images_list=glob.glob(os.path.join(image_dir,'*.png'))
    images_list = sorted(images_list)
    right = 0

    f=open(answer_txt,'w')

    for image_path in images_list:
        im=read_image(image_path,resize_height,resize_width,normalization=True)
        im=im[np.newaxis,:]
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pre_score,pre_label = sess.run([score,class_id], feed_dict={input_images:im})
        f.write(image_path + ': ' + label_to_char[int(pre_label)] + '\n')

        # max_score=pre_score[0,pre_label]

        print(image_path, label_to_char[int(pre_label)])
        if int(pre_label)== chartolabel[image_path.split('_')[-1].split('.')[0]]:
            right+=1
    acc = right/len(images_list)
    print('right: {}, test accuracy is: {}'.format(right,acc))
    f.write('right: {}, test accuracy is: {}'.format(right,acc))

    f.close()
    sess.close()



if __name__ == '__main__':

    class_nums= 62
    scheme = 'wiki'
    model = 'resnet'
    image_dir='dataset/'+ scheme + '/test'
    # models_path='models/'+model+'/singleChar2/model.ckpt-100000'
    models_path='models/'+model+'/fintune10000/'+scheme+'/model.ckpt-4000'

    batch_size = 1
    resize_height = 224
    resize_width = 224
    depths=3
    data_format=[batch_size,resize_height,resize_width,depths]
    answer_txt = 'test_results/' + model + '/'+scheme+'_'+model+'_answer_random_fintune10000.txt'

    predict(models_path,image_dir, class_nums, data_format,model, answer_txt)
