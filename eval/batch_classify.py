
import os
import sys

sys.path.append((os.path.normpath(
                 os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..'))))

import argparse
import glob
import tensorflow as tf
from tqdm import tqdm

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader


IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def create_batch_iterator(filenames, batch_size, fn_load_image):
    for i in range(0, len(filenames), batch_size):
        yield list(map(fn_load_image, filenames[i:i+batch_size]))


def create_tf_batch_iterator(filenames, batch_size):
    for i in range(0, len(filenames), batch_size):
        with tf.Session(graph=tf.Graph()) as session:
            fn_load_image = create_tensorflow_image_loader(session,
                                                           expand_dims=False)

            yield list(map(fn_load_image, filenames[i:i+batch_size]))


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source", required=True,
                        help="Folder containing the images to classify")

    parser.add_argument("-o", "--output_file", required=True,
                        help="Output file path")

    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-b", "--batch_size", help="Number of images to \
                        classify simultaneously.", type=int, default=64)

    parser.add_argument("-l", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    args = parser.parse_args()
    batch_size = args.batch_size
    output_file = args.output_file

    input_type = InputType.TENSOR
    model = OpenNsfwModel()

    filenames = glob.glob(args.source + "/*.jpg")
    num_files = len(filenames)

    num_batches = int(num_files / batch_size)

    print("Found", num_files, " files")
    print("Split into", num_batches, " batches")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    batch_iterator = None

    if args.image_loader == IMAGE_LOADER_TENSORFLOW:
        batch_iterator = create_tf_batch_iterator(filenames, batch_size)
    else:
        fn_load_image = create_yahoo_image_loader(expand_dims=False)
        batch_iterator = create_batch_iterator(filenames, batch_size,
                                               fn_load_image)

    with tf.Session(graph=tf.Graph(), config=config) as session:
        model.build(weights_path=args.model_weights,
                    input_type=input_type)

        session.run(tf.global_variables_initializer())

        with tqdm(total=num_files) as progress_bar:
            with open(output_file, 'w') as o:
                o.write('File\tSFW Score\tNSFW Score\n')

                for batch_num, images in enumerate(batch_iterator):
                    predictions = \
                        session.run(model.predictions,
                                    feed_dict={model.input: images})

                    fi = (batch_num * batch_size)
                    for i, prediction in enumerate(predictions):
                        filename = os.path.basename(filenames[fi + i])
                        o.write('{}\t{}\t{}\n'.format(filename,
                                                      prediction[0],
                                                      prediction[1]))

                    progress_bar.update(len(images))

if __name__ == "__main__":
    main(sys.argv)
