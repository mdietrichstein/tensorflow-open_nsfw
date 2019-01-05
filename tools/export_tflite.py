import os
import sys
import argparse

import tensorflow as tf

sys.path.append((os.path.normpath(
                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..'))))

from model import OpenNsfwModel, InputType

"""Exports a tflite version of tensorflow-open_nsfw

Note: The standard TFLite runtime does not support all required ops when using the base64_jpeg input type.
You will have to implement the missing ones by yourself.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="output filename, e.g. 'open_nsfw.tflite'")

    parser.add_argument("-i", "--input_type", required=True,
                        default=InputType.TENSOR.name.lower(),
                        help="Input type. Warning: base64_jpeg does not work with the standard TFLite runtime since a lot of operations are not supported",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    args = parser.parse_args()

    model = OpenNsfwModel()

    export_path = args.target
    input_type = InputType[args.input_type.upper()]

    with tf.Session() as sess:
        model.build(weights_path=args.model_weights,
                    input_type=input_type)

        sess.run(tf.global_variables_initializer())

        converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [model.input], [model.predictions])
        tflite_model = converter.convert()

        with open(export_path, "wb") as f:
            f.write(tflite_model)
