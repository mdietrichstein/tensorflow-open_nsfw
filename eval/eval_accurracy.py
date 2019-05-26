import sys
import operator
import argparse
import numpy as np
from scipy import stats


def load_classifications(filename):
    is_first = True

    results = {}

    with open(filename, 'r') as f:
        for line in f:
            if is_first:
                is_first = False
                continue

            parts = line.split('\t')

            filename = parts[0]
            sfw_score = float(parts[1])
            nsfw_score = float(parts[2])

            results[filename] = (sfw_score, nsfw_score)

    return results


def classification_matrix(classifications):
    results = np.zeros(shape=(len(classifications), 2))

    for i, classification in enumerate(classifications):
        results[i] = np.array(classification[1])

    return results


def test_acc(nsfw):
    count = len(nsfw)
    result = {
        'sfw': (nsfw < 0.5).sum() / count,
        'nsfw': (nsfw >= 0.7).sum() / count,
        'sexy': ((nsfw >= 0.5) & (nsfw < 0.7)).sum() / count,
    }

    return result


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input",
                        help="File containing classifications")

    args = parser.parse_args()
    filename_input = args.input

    result = load_classifications(filename_input)

    result = sorted(result.items(), key=operator.itemgetter(0))

    print("Found", len(result), "entries")

    classifications = classification_matrix(result)

    result = test_acc(classifications[:, 1]);
    print(result)


if __name__ == "__main__":
    main(sys.argv)
