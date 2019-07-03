
import os
import sys
import tensorflow as tf
import tensorflow.contrib.tensorrt

import argparse

sys.path.append((os.path.normpath(
                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..'))))

from image_utils import create_yahoo_image_loader
from graph_utils import get_frozen_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help="Path to the input image.\
                            Only jpeg images are supported.")

    parser.add_argument("-m", "--frozen_graph_file", required=True,
                        help="The frozen graph def file")

    args = parser.parse_args()

    input_node_name = 'input'
    output_node_name = 'predictions'

    trt_graph = get_frozen_graph(args.frozen_graph_file)

    # Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(trt_graph, name='')

    # input and output tensor names.
    input_tensor_name = input_node_name + ":0"
    output_tensor_name = output_node_name + ":0"

    print("input_tensor_name: {}\noutput_tensor_name: {}\n".format(
          input_tensor_name, output_tensor_name))

    output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

    fn_load_image = create_yahoo_image_loader()

    img = fn_load_image(args.input_file)

    feed_dict = {
        input_tensor_name: img
    }
    preds = tf_sess.run(output_tensor, feed_dict)

    print("Results for '{}'".format(args.input_file))
    print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*preds[0]))

