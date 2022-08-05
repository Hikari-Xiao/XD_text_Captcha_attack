"""

"""

import os
import numpy as np
import tensorflow as tf
import cv2

# +-* + () + 10 digit + blank + space
#charset +blank +space 字符数+2
num_classes =27+2 #edit

maxPrintLen = 4 #edit
                # 5:360,360_gray,apple
                # 4:alipay,baidu,baidu_blue,baidu_red,jd,jd_grey,jd_white,qqmail,weibo
                # 6:ms
                # 10:wiki

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/smalldata_test', 'the checkpoint dir') #edit
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 53, 'image height') #edit
tf.app.flags.DEFINE_integer('image_width', 150, 'image width') #edit
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')
tf.app.flags.DEFINE_integer('max_stepsize', 4, 'max lenth of sequence')#edit

tf.app.flags.DEFINE_integer('cnn_count', 4, 'count of cnn module to extract image features.')
tf.app.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 2000, 'maximum epochs') #edit
tf.app.flags.DEFINE_integer('batch_size', 40, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 2000, 'the step to save checkpoint')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_integer('validation_steps', 500, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam opt,,,,,,,,,imizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir', '/home/abc/xcx/CRNNdata/baidu_blue_/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', '/home/abc/xcx/crnn_500/baidu_blue_/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', '/home/abc/xcx/crnn_500/baidu_blue_/test/', 'the infer data dir') #edit
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'infer', 'train, val or infer')#edit
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')

FLAGS = tf.app.flags.FLAGS

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)

#charset = 'abcdefghijklmnopqrstuvwxyz' # wiki 26
#charset = '2345678abdefghjmnqruyABCDEFGHJKMNPQRSUVWXYZ'  # 360 43
#charset = '2345678abdefghjmnqruyABCDEFGHJKMNPQRSUVWXYZ'  # 360_gray 43
#charset = '345678ABCEFGHJKMNPQRSUWXY'     # weibo 25
#charset = '2347abdeghmnqrtyABCDEFGHIJKLMNOPQRSTUVWXYZ'  # qqmail 42
#charset = '34568ABCEFHKMNRSTUVWXY'  # jd/jd_grey/jd_white 22
# charset = '2345678abdefhmnqyABCEFGHKMNPQRSUVWXYZ'  # sina
#charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # baidu 26
#charset = 'ABCDEFGHIJKLMNOPQRSTUVWXY'  #  baidu_red 25
#charset = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # alipay 36
#charset = '3456DGHJKLMNPQRSVWXYdy'  # ms 22
#charset = '123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'  # apple 34
charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYuy'  # baidu_blue 27

encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char
print(encode_maps)

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN
print(encode_maps)


class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            # print(root, sub_folder)
            for file_path in file_list:
                # print('file path-----'+file_path)
                image_name = os.path.join(root, file_path)
                #print('image name======'+image_name)
                im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
                # resize to same height, different width will consume time on padding
                im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height))
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                self.image.append(im)

                # image is named as /.../<folder>/00000_abcd.png
                code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
                self.labels.append(code)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            # 64 is the output channels of the last layer of CNN
            lengths = np.asarray([FLAGS.out_channels for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open('./test.csv', 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs
