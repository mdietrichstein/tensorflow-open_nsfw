import os
import sys
import argparse

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io


sys.path.append((os.path.normpath(
                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..'))))

from model import OpenNsfwModel, InputType


def freeze_graph(graph, session, output):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)

        return graphdef_frozen


"""Exports the graph so it can be imported via import_graph_def

The exported model takes an base64 encoded string tensor as input
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="output directory")

    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-i", "--input_type", required=False,
                        default=InputType.TENSOR.name.lower(),
                        help="Input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    args = parser.parse_args()

    model = OpenNsfwModel()

    export_base_path = args.target
    do_freeze = True
    input_type = InputType[args.input_type.upper()]

    input_node_name = 'input'
    output_node_name = 'predictions'

    base_name = 'open_nsfw'

    checkpoint_path = os.path.join(export_base_path, base_name + '.ckpt')

    graph_name = base_name + '.pb'

    graph_path = os.path.join(export_base_path, graph_name)

    with tf.Session() as sess:
        model.build(weights_path=args.model_weights,
                    input_type=input_type)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.save(sess, save_path=checkpoint_path)

        print('Checkpoint exported to {}'.format(checkpoint_path))

        tf.train.write_graph(sess.graph_def, export_base_path, graph_name,
                             as_text=False)

        print('Graph exported to {}'.format(graph_path))

        print('Freezing graph...')
        frozen_graph_def = freeze_graph(sess.graph, sess, [output_node_name])

        trt_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph_def,
                outputs=[output_node_name],
                max_batch_size=1,
                max_workspace_size_bytes=1 << 25,
                precision_mode='FP16',
                minimum_segment_size=50
        )

        graph_io.write_graph(trt_graph, export_base_path, 'trt_' + graph_name, as_text=False)
