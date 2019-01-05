import base64
import json
import argparse

from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS

"""base64 encodes the given input jpeg and outputs json data suitable for
'gcloud ml-engine predict' requests to a model generated with 'export-model.py'
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input image file")
    parser.add_argument("-t", "--target", required=True,
                        choices=['ml-engine', 'tf-serving'],
                        help="Create json for ml-engine or tensorflow-serving")

    args = parser.parse_args()
    target = args.target

    image_b64 = base64.urlsafe_b64encode(open(args.input_file, "rb").read())

    if target == "ml-engine":
        print(json.dumps({PREDICT_INPUTS: image_b64.decode("ascii")}))
    elif target == "tf-serving":
        print(json.dumps({"instances": [image_b64.decode("ascii")]}))
