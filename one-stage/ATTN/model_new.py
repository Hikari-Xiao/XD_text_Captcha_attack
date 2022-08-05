import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags
from tensorflow.contrib.slim.nets import inception
import sequence_nn
from pprint import pprint
FLAGS = flags.FLAGS
class Attention_Model(object):
    def __init__(self,data,label):
        super(Attention_Model, self).__init__()
        self._data= data
        self._label = label

    def pool_views_fn(self, net):
        with tf.variable_scope('pool_views_fn/STCK'):
            batch_size = net.get_shape().dims[0].value
            feature_size = net.get_shape().dims[3].value
            res = tf.reshape(net, [batch_size, -1, feature_size])
            # print res.get_shape()
            return res

    def conv_tower_fn(self, images, is_training=True, reuse=None):
        with tf.variable_scope('conv_tower_fn/INCE'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                net, _ = inception.inception_v3_base(
                    images, final_endpoint=FLAGS.final_endpoint)
            return net

    def create_model(self,scope='AttentionOcr',
                  reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            print (self._data)
            net = self.conv_tower_fn(self._data, FLAGS.is_training)
            print net.get_shape()
            net = self.pool_views_fn(net)
            seq_net = sequence_nn.SequenceLayerBase(net,self._label)#
            chars_logit = seq_net.create_logits()
            self.out_attns = seq_net.out_attns
            # chars_logit = tf.reshpe(chars_logit,[chars_logit.get_shape().dimp[0].value,-1])
            # chars_logit = slim.fully_connected(chars_logit,FLAGS.img_height*FLAGS.img_width)
            # self.predict = tf.reshape(chars_logit,[-1,FLAGS.img_height,FLAGS.img_width])
            # print chars_logit.get_shape()
            self.predict = tf.argmax(chars_logit,axis=-1)
            self.sequence_loss_fn(chars_logit,self._label)
            total_loss = slim.losses.get_total_loss()
            # pprint(slim.losses.get_regularization_losses())
            self.loss = total_loss

    def sequence_loss_fn(self, chars_logits, chars_labels):

        with tf.variable_scope('sequence_loss_fn/SLF'):
            if FLAGS.label_smoothing > 0:
                smoothed_one_hot_labels = self.label_smoothing_regularization(chars_labels = chars_labels, weight=FLAGS.label_smoothing)
                labels_list = tf.unstack(smoothed_one_hot_labels, axis=1)
            else:
                # NOTE: in case of sparse softmax we are not using one-hot
                # encoding.
                labels_list = tf.unstack(chars_labels, axis=1)

            batch_size, seq_length, _ = chars_logits.shape.as_list()
            if FLAGS.ignore_nulls:
                weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
            else:
                # Suppose that reject character is the last in the charset.
                reject_char = tf.constant(
                    self._params.num_char_classes - 1,
                    shape=(batch_size, seq_length),
                    dtype=tf.int64)
                known_char = tf.not_equal(chars_labels, reject_char)
                weights = tf.to_float(known_char)

            logits_list = tf.unstack(chars_logits, axis=1)
            weights_list = tf.unstack(weights, axis=1)
            loss = tf.contrib.legacy_seq2seq.sequence_loss(
                logits_list,
                labels_list,
                weights_list,
                softmax_loss_function=self.get_softmax_loss_fn(FLAGS.label_smoothing),
                average_across_timesteps=FLAGS.average_across_timesteps)
            tf.losses.add_loss(loss)
            return loss

    def label_smoothing_regularization(self,chars_labels, weight=0.1):
        # print type(FLAGS.num_char_classes)
        # print FLAGS.num_char_classes
        pos_weight = 1.0 - weight
        neg_weight = weight / FLAGS.num_char_classes
        return chars_labels * pos_weight + neg_weight

    def get_softmax_loss_fn(self,label_smoothing):
        """Returns sparse or dense loss function depending on the label_smoothing.

          Args:
            label_smoothing: weight for label smoothing

          Returns:
            a function which takes labels and predictions as arguments and returns
            a softmax loss for the selected type of labels (sparse or dense).
          """
        if label_smoothing > 0:

            def loss_fn(labels, logits):
                return (tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))
        else:

            def loss_fn(labels, logits):
                return tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)

        return loss_fn

