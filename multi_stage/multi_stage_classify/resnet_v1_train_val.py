#coding=utf-8

import tensorflow as tf 
import numpy as np 
import pdb
import os
from datetime import datetime
import slim.nets.resnet_v1 as resnet_v1
from create_tf_record import *
import tensorflow.contrib.slim as slim

labels_nums = 62
batch_size = 8
resize_height = 224
resize_width = 224
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')

def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, keep_prob:1.0, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc

def step_train(train_op,loss,accuracy,
               train_images_batch,train_labels_batch,train_nums,train_log_step,
               val_images_batch,val_labels_batch,val_nums,val_log_step,
               snapshot_prefix,snapshot):
    saver = tf.train.Saver(max_to_keep=10)
    max_acc = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, "models/resnet/singleChar2/model.ckpt-100000")

        for i in range(max_steps + 1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images: batch_input_images,
                                                                  input_labels: batch_input_labels,
                                                                  keep_prob: 0.8, is_training: True})
            if i % train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images: batch_input_images,
                                                          input_labels: batch_input_labels,
                                                          keep_prob: 1.0, is_training: False})
                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (
                datetime.now(), i, train_loss, train_acc))

            if i % val_log_step == 0:
                mean_loss, mean_acc = net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch, val_nums)
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))

            if (i % snapshot == 0 and i > 0) or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix, i))
                saver.save(sess, snapshot_prefix, global_step=i)
            if mean_acc > max_acc and mean_acc > 0.7:
                max_acc = mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models = os.path.join(path, 'best_models_{}_{:.4f}.ckpt'.format(i, max_acc))
                print('------save:{}'.format(best_models))
                saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)

def train(train_record_file,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          data_shape,
          snapshot,
          snapshot_prefix):
    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    train_nums=get_example_nums(train_record_file)
    val_nums=get_example_nums(val_record_file)
    print('train nums:%d,val nums:%d'%(train_nums,val_nums))

    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization')
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=True)
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization')
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)

    # Define the model:
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        out, end_points = resnet_v1.resnet_v1_50(inputs=input_images, num_classes=labels_nums, is_training=is_training,global_pool=True)
    # with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
    #     out, end_points = mobilenet_v1.mobilenet_v1(inputs=input_images, num_classes=labels_nums,
    #                                                 dropout_keep_prob=keep_prob, is_training=is_training,
    #                                                 global_pool=True)

    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)

    # Specify the optimization scheme:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        print("update_ops:{}".format(update_ops))
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = tf.train.MomentumOptimizer(learning_rate=base_lr, momentum=0.9).minimize(loss)
        train_op = tf.train.AdadeltaOptimizer(learning_rate=base_lr).minimize(loss)


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
    step_train(train_op=train_op, loss=loss, accuracy=accuracy,
               train_images_batch=train_images_batch,
               train_labels_batch=train_labels_batch,
               train_nums=train_nums,
               train_log_step=train_log_step,
               val_images_batch=val_images_batch,
               val_labels_batch=val_labels_batch,
               val_nums=val_nums,
               val_log_step=val_log_step,
               snapshot_prefix=snapshot_prefix,
               snapshot=snapshot)


if __name__ == '__main__':
    scheme = 'wiki'
    train_record_file = 'dataset/fintune10000/' + scheme + '_train224.tfrecords'
    val_record_file = 'dataset/fintune10000/' + scheme + '_val224.tfrecords'

    train_log_step=100
    base_lr = 0.01

    max_steps = 4000
    train_param=[base_lr,max_steps]

    val_log_step=1000
    snapshot=2000
    model_file = "models/resnet/fintune10000/"+scheme
    if not os.path.exists(model_file):
        os.mkdir(model_file)
    snapshot_prefix='models/resnet/fintune10000/'+scheme+'/model.ckpt'
    train(train_record_file=train_record_file,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
