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


def test(first, second):
    delta = np.abs(first - second)

    result = {
        'min': np.amin(delta),
        'max': np.amax(delta),
        'median': np.median(delta),
        'mean': np.mean(delta),
        'std': np.std(delta),
        'var': np.var(delta),
        't-prob': stats.ttest_ind(first, second, equal_var=True)[1]
    }

    return result


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("original",
                        help="File containing base classifications")

    parser.add_argument("other",
                        help="File containing classifications to compare to\
                        base results")

    args = parser.parse_args()
    filename_original = args.original
    filename_other = args.other

    original = load_classifications(filename_original)
    other = load_classifications(filename_other)

    len(original) == len(other)

    original = sorted(original.items(), key=operator.itemgetter(0))
    other = sorted(other.items(), key=operator.itemgetter(0))

    print("Found", len(original), "entries")

    original_classifications = classification_matrix(original)
    other_classifications = classification_matrix(other)

    print('SFW:')
    print(test(original_classifications[:, 0], other_classifications[:, 0]))

    print()
    print('NSFW:')
    print(test(original_classifications[:, 1], other_classifications[:, 1]))

if __name__ == "__main__":
    main(sys.argv)
