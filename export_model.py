import os
import argparse

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils\
    import predict_signature_def

from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.signature_constants\
    import DEFAULT_SERVING_SIGNATURE_DEF_KEY

from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS

from model import OpenNsfwModel, InputType

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("target", help="output directory")

    parser.add_argument("-v", "--export_version",
                        help="export model version",
                        default="1")

    parser.add_argument("-m", "--model_weights", required=True,
                        help="Trained model weights file")

    args = parser.parse_args()

    model = OpenNsfwModel()

    export_base_path = args.target
    export_version = args.export_version

    export_path = os.path.join(export_base_path, export_version)

    with tf.Session() as sess:
        model.build(weights_path=args.model_weights,
                    input_type=InputType.BASE64_JPEG)

        sess.run(tf.global_variables_initializer())

        builder = saved_model_builder.SavedModelBuilder(export_path)

        builder.add_meta_graph_and_variables(
            sess, [SERVING],
            signature_def_map={
                DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def(
                    inputs={PREDICT_INPUTS: model.input},
                    outputs={PREDICT_OUTPUTS: model.predictions}
                )
            }
        )

        builder.save()
