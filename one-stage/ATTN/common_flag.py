#coding:utf-8
from tensorflow.python.platform import flags
#tf定义了tf.app.flags，用于支持接受命令行传递参数，相当于接受argv
#
FLAGS = flags.FLAGS#flags是一个文件：flags.py，用于处理命令行参数的解析工作
def define():
    flags.DEFINE_float("weight_decay",0.00004,"weight_decay")#第一个是参数名称，第二个参数是默认值，第三个是参数描述
    flags.DEFINE_float("learning_rate", 0.006, "learning_rate")
    flags.DEFINE_float("momentum", 0.9, "momentum")
    flags.DEFINE_float("clip_gradient_norm", 2.0, "clip_gradient_norm")
    flags.DEFINE_integer("num_char_classes",745,"number of char classes")
    #charclasses+1(这个+1至关重要！！！！！，不然loss会一直增大)
    flags.DEFINE_bool("is_training",False,"is training")
    flags.DEFINE_integer("seq_length",4, "")#训练集的长度
    flags.DEFINE_integer('num_lstm_units', 256,
                         'number of LSTM units for sequence LSTM')
    flags.DEFINE_float('lstm_state_clip_value', 10.0,
                       'cell state is clipped by this value prior to the cell'
                       ' output activation')
    flags.DEFINE_float('label_smoothing', 0.1,
                       'weight for label smoothing')
    flags.DEFINE_string("final_endpoint","Mixed_6d",'Endpoint to cut inception tower')
    flags.DEFINE_integer("img_width", 300, "")
    flags.DEFINE_integer("img_height", 100, "")
    flags.DEFINE_integer("attn_width", 17, "")
    flags.DEFINE_integer("attn_height", 4, "")
    flags.DEFINE_integer("img_channel", 3, "")
    flags.DEFINE_bool("ignore_nulls",True,"see nulls as normal label")
    flags.DEFINE_bool("average_across_timesteps",False,"LSTM config not know")
