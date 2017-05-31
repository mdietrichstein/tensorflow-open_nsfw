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

    args = parser.parse_args()

    image_b64 = base64.urlsafe_b64encode(open(args.input_file, "rb").read())

    print(json.dumps({PREDICT_INPUTS: image_b64.decode("ascii")}))
