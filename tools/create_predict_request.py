import base64
import json
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS

import os
import sys

sys.path.append((os.path.normpath(
                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..'))))

from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
from model import InputType

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

# Thanks to https://stackoverflow.com/a/47626762
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

"""Generates a json prediction request suitable for consumption by a model 
generated with 'export-model.py' and deployed on either ml-engine or tensorflow-serving
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input image file")

    parser.add_argument("-i", "--input_type", required=True,
                        default=InputType.TENSOR.name.lower(),
                        help="Input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    parser.add_argument("-l", "--image_loader", required=False,
                        default=IMAGE_LOADER_YAHOO,
                        help="Image loading mechanism. Only relevant when using input_type 'tensor'",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--target", required=True,
                        choices=['ml-engine', 'tf-serving'],
                        help="Create json request for ml-engine or tensorflow-serving")

    args = parser.parse_args()
    target = args.target

    input_type = InputType[args.input_type.upper()]

    image_data = None

    if input_type == InputType.TENSOR:
        fn_load_image = None

        if args.image_loader == IMAGE_LOADER_TENSORFLOW:
            with tf.Session() as sess:
                fn_load_image = create_tensorflow_image_loader(sess)
                sess.run(tf.global_variables_initializer())
                image_data = fn_load_image(args.input_file)[0]
        else:
            image_data = create_yahoo_image_loader(tf.Session(graph=tf.Graph()))(args.input_file)[0]
    elif input_type == InputType.BASE64_JPEG:
        import base64
        image_data = base64.urlsafe_b64encode(open(args.input_file, "rb").read()).decode("ascii")

    if target == "ml-engine":
        print(json.dumps({PREDICT_INPUTS: image_data}, cls=NumpyEncoder))
    elif target == "tf-serving":
        print(json.dumps({"instances": [image_data]}, cls=NumpyEncoder))
