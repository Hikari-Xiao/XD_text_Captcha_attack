#coding:utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
from read_tfrecord import *
import cv2
import numpy as np
import sys
import model
import os
# from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell_impl import LSTMCell
import common_flag
from tensorflow.python.platform import flags
from pprint import pprint
import tensorflow.contrib.deprecated as tfsummary
FLAGS = flags.FLAGS
common_flag.define()
#model parameter
batch_size = 5
IMG_WIDTH = FLAGS.img_width
IMG_HEIGHT = FLAGS.img_height
IMG_CHANNEL = FLAGS.img_channel
WEIGHT_DECAY = FLAGS.weight_decay
num_classes = FLAGS.num_char_classes
epoch = 10000
learning_rate = FLAGS.learning_rate
iteration_in_epoch = 1000
iteration_in_test = 100

#--------------------------------------------------------------------
#data
net_input = tf.placeholder("float",[batch_size,FLAGS.img_height,FLAGS.img_width,FLAGS.img_channel])
net_image_label = tf.placeholder("float",[batch_size,FLAGS.seq_length,FLAGS.num_char_classes])
# net_learning = tf.placeholder("float",None)
#model
attention_model = model.Attention_Model(net_input,net_image_label)
attention_model.create_model()
# attention_out = attention_model.out_attns
#img_loss
img_loss = attention_model.loss
net_predict = attention_model.predict
net_accuracy = tf.cast(tf.equal(net_predict,tf.arg_max(net_image_label,-1)),tf.float32)
net_acc = tf.reduce_sum(net_accuracy, -1)
net_accuracy = tf.reduce_mean(net_accuracy)
net_acc = tf.cast(tf.equal(net_acc, FLAGS.seq_length), tf.float32)
net_acc = tf.reduce_mean(net_acc)
# ---------------------------------------------
#all loss
net_loss = img_loss
net_optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate,FLAGS.momentum)
net_train = slim.learning.create_train_op(net_loss,net_optimizer,summarize_gradients=True,clip_gradient_norm=FLAGS.clip_gradient_norm)
# tst_loss_l = slim.losses.get_regularization_losses()
# tst_loss = tst_loss_l[0]
# for i in range(1,len(tst_loss_l)):
#     tst_loss+=tst_loss_l[i]

#initialize weight
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
# summarys:
all_variables = tf.global_variables()
for v in all_variables:
    tf.summary.histogram(v.name,v)
tf.summary.image("train_images",net_input,10)
tf.summary.scalar("net_loss",net_loss)
tf.summary.scalar("net_accuracy",net_accuracy)
tf.summary.scalar("net_acc",net_acc)
merge_summary = tfsummary.merge_all_summaries()
#summary_writer = tf.summary.FileWriter("train_log/",sess.graph)
saver.restore(sess,"model/500fine/jd/attention_5d_best_0.3169140625.data")

#load data
file_dir = "/home/abc/xcx/CRNN/CRNNdata/jd/test/"
files_paths = os.listdir(file_dir)
files_paths.sort()
#from chineseMap_QQ import label_to_char # for chinese
label_to_char ="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
answer_file = open("answer.txt", 'w')
right = 0 # record test set right number
imgs = []
labels = []
num = 0
for p in files_paths:
    # print p
    num += 1
    img = cv2.imread(file_dir+p)
    #tmp_chrs = p.split("_")[1].split(".")[0]
    #tmp_labels = [chartolabel[c] for c in tmp_chrs]
    #while len(tmp_labels) != Seq_length:
        #tmp_labels.append(Fill_label)
    img = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT))
    # img = np.asarray(img, np.uint8)
    img = img / 255.
    img = img - 0.5
    img = img * 2.5
    imgs.append(img)
    labels.append(p)
    if len(imgs)%batch_size == 0:
        gen_imgs = imgs
        predictions = sess.run(net_predict,feed_dict={
                net_input : gen_imgs})

        for b in range(batch_size):
            tmp_index = predictions[b]
            tmp_chars = ""
            for index in tmp_index:
                tmp_chars += str(index)
                tmp_chars += label_to_char[index]
            print(tmp_chars,labels[b])
            if (tmp_chars == labels[b].split('_')[1].split('.')[0]):
                right += 1
            answer_file.writelines(labels[b] + ":" + str(tmp_chars) + '\n')
        imgs = []
        labels = []
print("test accuracy:" + str(float(right)/(len(files_paths) - len(files_paths) % batch_size)))
answer_file.writelines("test accuracy:" + str(float(right)/(len(files_paths) - len(files_paths)%batch_size)) + '\n')
