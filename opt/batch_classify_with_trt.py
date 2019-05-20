
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
from image_utils import create_yahoo_image_loader
from graph_utils import get_frozen_graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


def create_batch_iterator(filenames, batch_size, fn_load_image):
    for i in range(0, len(filenames), batch_size):
        yield list(map(fn_load_image, filenames[i:i+batch_size]))


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source", required=True,
                        help="Folder containing the images to classify")

    parser.add_argument("-o", "--output_file", required=True,
                        help="Output file path")

    parser.add_argument("-m", "--frozen_graph_file", required=True,
                        help="The frozen graph def file")

    parser.add_argument("-b", "--batch_size", help="Number of images to \
                        classify simultaneously.", type=int, default=64)

    args = parser.parse_args()
    batch_size = args.batch_size
    output_file = args.output_file

    filenames = glob.glob(args.source + "/*.jpg")
    num_files = len(filenames)

    num_batches = int(num_files / batch_size)

    print("Found", num_files, " files")
    print("Split into", num_batches, " batches")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    fn_load_image = create_yahoo_image_loader(expand_dims=False)
    batch_iterator = create_batch_iterator(filenames, batch_size, fn_load_image)

    trt_graph = get_frozen_graph(args.frozen_graph_file)
    # Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(trt_graph, name='')

    input_node_name = 'input'
    output_node_name = 'predictions'

    # input and output tensor names.
    input_tensor_name = input_node_name + ":0"
    output_tensor_name = output_node_name + ":0"

    print("input_tensor_name: {}\noutput_tensor_name: {}\n".format(
          input_tensor_name, output_tensor_name))

    output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

    with tqdm(total=num_files) as progress_bar:
        with open(output_file, 'w') as o:
            o.write('File\tSFW Score\tNSFW Score\n')

            for batch_num, images in enumerate(batch_iterator):
                predictions = \
                    tf_sess.run(output_tensor,
                                feed_dict={input_tensor_name: images})

                fi = (batch_num * batch_size)
                for i, prediction in enumerate(predictions):
                    filename = os.path.basename(filenames[fi + i])
                    o.write('{}\t{}\t{}\n'.format(filename,
                                                  prediction[0],
                                                  prediction[1]))

                progress_bar.update(len(images))


if __name__ == "__main__":
    main(sys.argv)
