import os
import sys
import argparse

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

sys.path.append((os.path.normpath(
                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..'))))

from model import OpenNsfwModel, InputType

"""Exports the graph so it can be imported via import_graph_def

The exported model takes an base64 encoded string tensor as input
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="output directory")

    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-o", "--optimize", required=False, default=True,
                        help="Optimize graph for inference")

    parser.add_argument("-f", "--freeze", required=False, default=True,
                        help="Freeze graph: convert variables to ops")

    parser.add_argument("-b", "--binary", required=False, default=True,
                        help="Write graph as binary (.pb) or text (pbtext)")

    args = parser.parse_args()

    model = OpenNsfwModel()

    export_base_path = args.target
    do_freeze = args.freeze
    do_optimize = args.optimize
    as_binary = args.binary

    input_node_name = 'input'
    output_node_name = 'predictions'

    base_name = 'open_nsfw'

    checkpoint_path = os.path.join(export_base_path, base_name + '.ckpt')

    if as_binary:
        graph_name = base_name + '.pb'
    else:
        graph_name = base_name + '.pbtxt'

    graph_path = os.path.join(export_base_path, graph_name)
    frozen_graph_path = os.path.join(export_base_path,
                                     'frozen_' + graph_name)
    optimized_graph_path = os.path.join(export_base_path,
                                        'optimized_' + graph_name)

    with tf.Session() as sess:
        model.build(weights_path=args.model_weights,
                    input_type=InputType.BASE64_JPEG)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.save(sess, save_path=checkpoint_path)

        print('Checkpoint exported to {}'.format(checkpoint_path))

        tf.train.write_graph(sess.graph_def, export_base_path, graph_name,
                             as_text=not as_binary)

        print('Graph exported to {}'.format(graph_path))

        if do_freeze:
            print('Freezing graph...')
            freeze_graph.freeze_graph(
                input_graph=graph_path, input_saver='',
                input_binary=as_binary, input_checkpoint=checkpoint_path,
                output_node_names=output_node_name,
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0',
                output_graph=frozen_graph_path, clear_devices=True,
                initializer_nodes='')

            print('Frozen graph exported to {}'.format(frozen_graph_path))

            graph_path = frozen_graph_path

        if do_optimize:
            print('Optimizing graph...')
            input_graph_def = tf.GraphDef()

            with tf.gfile.Open(graph_path, 'rb') as f:
                data = f.read()
                input_graph_def.ParseFromString(data)

                output_graph_def =\
                    optimize_for_inference_lib.optimize_for_inference(
                        input_graph_def,
                        [input_node_name],
                        [output_node_name],
                        tf.float32.as_datatype_enum)

                f = tf.gfile.FastGFile(optimized_graph_path, 'wb')
                f.write(output_graph_def.SerializeToString())

                print('Optimized graph exported to {}'
                      .format(optimized_graph_path))
