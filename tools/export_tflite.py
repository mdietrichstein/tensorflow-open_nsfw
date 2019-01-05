import os
import sys
import argparse

import tensorflow as tf

sys.path.append((os.path.normpath(
                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..'))))

from model import OpenNsfwModel, InputType

"""Exports a tflite version of tensorflow-open_nsfw

The exported model takes an base64 encoded string tensor as input.

Note: The standard TFLite runtime does not support all required ops.
You will have to implement the missing ones by yourself.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="output filename, e.g. 'open_nsfw.tflite'")

    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    args = parser.parse_args()

    model = OpenNsfwModel()

    export_path = args.target

    with tf.Session() as sess:
        model.build(weights_path=args.model_weights,
                    input_type=InputType.BASE64_JPEG)

        sess.run(tf.global_variables_initializer())

        converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [model.input], [model.predictions])
        tflite_model = converter.convert()

        with open(export_path, "wb") as f:
            f.write(tflite_model)
