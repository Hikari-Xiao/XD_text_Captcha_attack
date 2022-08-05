#coding:utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
from read_tfrecord import *
from glob import glob
import cv2
import numpy as np
import sys
import model
import os
import time

os.environ['CUDA_VISIBLE_DEVICES']='1'

# type_list = ['360', '360_gray', 'alipay', 'apple', 'baidu', 'baidu_blue', 'baidu_red',
#              'jd', 'jd_grey', 'jd_white', 'ms', 'qqmail', 'sina', 'weibo', 'wiki']
type_list = ['baidu_Chinese', 'dajie',  'douban', 'renmin','it168','random']
types = type_list[4]
# num_list = ['500','1000','2000','8500','fine','500fine','1000fine','2000fine','8500fine']
num_list = ['2w','4w','6w','8w','10w','fine','2w_fine','4w_fine','6w_fine','8w_fine','10w_fine']
nums = num_list[2]

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
epoch = 25
learning_rate = FLAGS.learning_rate
iteration_in_epoch = 1000
iteration_in_test = int(500/5)

def fun(types):
    paths = glob(os.path.join("/home/abc/LAB_workspace/attention_cap/model/chinese/{0}/{1}/".format(nums, types), '*'))
    best = 0.0
    for pp in paths:
        p = os.path.basename(pp)
        if p != "checkpoint":
            a = p.split('.')[0].split('_')[2]
            if a == "best":
                b = p.split('.')[0].split('_')[3]+'.'+p.split('.')[1]
                if float(b) > best:
                    best = float(b)
    return str(best)

#--------------------------------------------------------------------
#data
net_input = tf.placeholder("float",[batch_size,FLAGS.img_height,FLAGS.img_width,FLAGS.img_channel])
net_image_label = tf.placeholder("float",[batch_size,FLAGS.seq_length,FLAGS.num_char_classes])
net_learning = tf.placeholder("float",None)
#model
attention_model = model.Attention_Model(net_input,net_image_label)
attention_model.create_model()
#attention_model.out_attns
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
model_ac = fun(types)
print(model_ac)
# summary_writer = tf.summary.FileWriter("train_log/train_log_chinese1",sess.graph)
saver.restore(sess,"model/chinese/{0}/{1}/attention_5d_best_{2}.data".format(nums,types,model_ac))
#saver.restore(sess,"model/chinese/random/attention_5d_best_0.9959375.data")

#load data
data_test_img,data_test_labels = net_read_data("tfrecords/chinese/{0}/test.record".format(types),
                                    num_classes,IMG_WIDTH,IMG_HEIGHT,IMG_CHANNEL,is_train_set=False)
batch_test_imgs,batch_test_labels = tf.train.shuffle_batch([data_test_img,data_test_labels],batch_size,1000,100,4)

theads = tf.train.start_queue_runners(sess)
print("-----------------------"+types+" "+nums+"----------------------")
#test_net
# tst = net_output
# res = sess.run(tst,feed_dict={net_input : test_imgs})
# res = np.asarray(res)
# print res.shape
# print res

#test model
t_loss = 0
t_acc = 0
t_all_acc = 0
label_to_char ="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
with open("test_answer.txt", 'a') as answer_file:

    start = time.time()

    for i in range(iteration_in_test):
        # print(i)
        gen_imgs,gen_labels = sess.run([batch_test_imgs,batch_test_labels])

        #仅输出结果时使用
        i_loss,i_acc,i_all_acc,predictions = sess.run([net_loss,net_accuracy,net_acc,net_predict],feed_dict={
            net_input : gen_imgs,net_image_label: gen_labels,net_learning:learning_rate
        })
        t_loss += i_loss
        t_acc += i_acc
        t_all_acc += i_all_acc

        #输出预测结果时使用
        # for t in range(batch_size):
        #     p_chars = ""
        #     p_char = gen_labels[t]
        #     pred = predictions[t]
        #     pre = ""
        #     for b in range(4):
        #         p_index = p_char[b]
        #         pt_index = 0
        #         for k in range(63):
        #             pt_index = k
        #             if p_index[k] == 1:
        #                 break
        #
        #         # pre += label_to_char[pred[b]]
        #         # p_chars += label_to_char[pt_index]
        #         pre = pre+","+str(pred[b])
        #         p_chars = p_chars + "," + str(pt_index)
        #     print(str(p_chars) + "------" + str(pre))
            # answer_file.write(str(p_chars) + "------" + str(pre) + '\n')

    end = time.time()

print("loss:{0},acc:{1},acc_all:{2},total_time:{3}".format(
    t_loss/(iteration_in_test),t_acc/(iteration_in_test),t_all_acc/(iteration_in_test)
    ,end-start))
