# -*-coding: utf-8 -*-

import tensorflow as tf
from create_tf_record import *
from tensorflow.python.framework import graph_util

resize_height = 299
resize_width = 299
depths = 3

def freeze_graph_test(pb_path, image_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")

            output_tensor_name = sess.graph.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")

            im=read_image(image_path,resize_height,resize_width,normalization=True)
            im=im[np.newaxis,:]
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            out=sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
                                                        input_keep_prob_tensor:1.0,
                                                        input_is_training_tensor:False})
            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            print("pre class_id:{}".format(sess.run(class_id)))


def freeze_graph(input_checkpoint,output_graph):
    output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        # for op in sess.graph.get_operations():
        #     print(op.name, op.values())

def freeze_graph2(input_checkpoint,output_graph):
    # checkpoint = tf.train.get_checkpoint_state(model_folder)
    # input_checkpoint = checkpoint.model_checkpoint_path

    output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

        # for op in graph.get_operations():
        #     print(op.name, op.values())


if __name__ == '__main__':
    input_checkpoint='models/model.ckpt-10000'
    out_pb_path="models/pb/frozen_model.pb"
    freeze_graph(input_checkpoint,out_pb_path)

    image_path = 'test_image/animal.jpg'
    freeze_graph_test(pb_path=out_pb_path, image_path=image_path)
