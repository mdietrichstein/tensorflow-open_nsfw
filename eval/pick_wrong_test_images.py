import os
import sys
import argparse


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input",
                        help="File containing wrong images")

    parser.add_argument("-s", "--source", required=True,
                        help="Folder containing the test images")

    parser.add_argument("-o", "--output", required=True,
                        help="Folder to contain the wrong label images")

    args = parser.parse_args()
    filename_input = args.input

    if os.path.exists(args.output) & (not os.path.isdir(args.output)):
        print("{} exists but not a directory".format(args.output))
        return
    elif not os.path.exists(args.output):
        os.mkdir(args.output)

    with open(filename_input, 'r') as f:
        for line in f:
            line = line.rstrip()
            source_file = os.path.join(args.source, line)
            dest_file = os.path.join(args.output, line)
            os.rename(source_file, dest_file)


if __name__ == "__main__":
    main(sys.argv)