import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags
import common_flag
FLAGS = flags.FLAGS

def orthogonal_initializer(shape, dtype=tf.float32, *args, **kwargs):
  del args
  del kwargs
  flat_shape = (shape[0], np.prod(shape[1:]))
  w = np.random.randn(*flat_shape)
  u, _, v = np.linalg.svd(w, full_matrices=False)
  w = u if u.shape == flat_shape else v
  return tf.constant(w.reshape(shape), dtype=dtype)

class SequenceLayerBase(object):
    def __init__(self,net,label):
        self._net = net
        self._label = label
        self._batch_size = net.get_shape()[0]
        self.is_training = FLAGS.is_training
        self._char_logits = {}
        regularizer = slim.l2_regularizer(FLAGS.weight_decay)
        self._softmax_w = slim.model_variable(
            'softmax_w',
            [FLAGS.num_lstm_units, FLAGS.num_char_classes],
            initializer=orthogonal_initializer,
            regularizer=regularizer)
        self._softmax_b = slim.model_variable(
            'softmax_b', [FLAGS.num_char_classes],
            initializer=tf.zeros_initializer(),
            regularizer=regularizer)
        self._zero_label = tf.zeros(
            [self._batch_size, FLAGS.num_char_classes])

    def char_one_hot(self, logit):
        """Creates one hot encoding for a logit of a character.

        Args:
          logit: A tensor with shape [batch_size, num_char_classes].

        Returns:
          A tensor with shape [batch_size, num_char_classes]
        """
        prediction = tf.argmax(logit, dimension=1)
        return slim.one_hot_encoding(prediction, FLAGS.num_char_classes)

    def get_train_input(self, prev, i):
        """See SequenceLayerBase.get_train_input for details."""
        if i == 0:
            return self._zero_label
        else:
            # TODO(gorban): update to gradually introduce gt labels.
            return self._label[:, i - 1, :]

    def get_eval_input(self, prev, i):
        """See SequenceLayerBase.get_eval_input for details."""
        if i == 0:
            return self._zero_label
        else:
            logit = self.char_logit(prev, char_index=i - 1)
            return self.char_one_hot(logit)

    def get_input(self, prev, i):
        if self.is_training:
            return self.get_train_input(prev, i)
        else:
            return self.get_eval_input(prev, i)

    def unroll_cell(self, decoder_inputs, initial_state, loop_function, cell):
        return tf.contrib.legacy_seq2seq.attention_decoder(
            decoder_inputs=decoder_inputs,
            initial_state=initial_state,
            attention_states=self._net,
            cell=cell,
            loop_function=self.get_input)

    def char_logit(self, inputs, char_index):
        if char_index not in self._char_logits:
            self._char_logits[char_index] = tf.nn.xw_plus_b(inputs, self._softmax_w,
                                                            self._softmax_b)
        return self._char_logits[char_index]
    def create_logits(self):
        with tf.variable_scope('LSTM'):
            first_label = self.get_input(prev=None, i=0)
            decoder_inputs = [first_label] + [None] * (FLAGS.seq_length - 1)
            lstm_cell = tf.contrib.rnn.LSTMCell(
                FLAGS.num_lstm_units,
                use_peepholes=False,
                cell_clip=FLAGS.lstm_state_clip_value,
                state_is_tuple=True,
                initializer=orthogonal_initializer)
            lstm_outputs, _ = self.unroll_cell(
                decoder_inputs=decoder_inputs,
                initial_state=lstm_cell.zero_state(self._batch_size, tf.float32),
                loop_function=self.get_input,
                cell=lstm_cell)

        with tf.variable_scope('logits'):
            logits_list = [
                tf.expand_dims(self.char_logit(logit, i), dim=1)
                for i, logit in enumerate(lstm_outputs)
                ]

        return tf.concat(logits_list, 1)