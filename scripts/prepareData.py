#! /usr/bin/env python
"""
"""

import argparse
import sys

import numpy as np

import utils.mnist as mnist


def parse_args(arg_list):
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", dest="data_dir", type=str, required=True,
            help="Directory containing the downloaded MNIST digits data.")
    parser.add_argument("-t", dest="data_type", type=str, required=True,
            help="Data type: 'training', 'test', or 'all'")
    parser.add_argument("-n", dest="num_to_keep", type=int, default=None, required=False,
            help="Read the first n digits.  Default: read all digits for the given data type.")
    parser.add_argument("-o", dest="out_base", type=str, required=True,
            help="The output file base: 1) <out_base>-data.tsv 2) <out_base>-labels.txt")

    args = parser.parse_args(arg_list)

    return args


def main():
    """
    """
    args = parse_args(sys.argv[1:])

    mnist_data_dir = args.data_dir
    dataset = args.data_type
    num_to_keep = args.num_to_keep

    # load data
    labels, images, vectors = [], [], []
    for label, img in mnist.read(dataset=dataset, path=mnist_data_dir):
        labels.append(label)
        images.append(img) 

        # each 25 x 25 array is transformed to a length 625 vector and normalized between 0.0 and 1.0
        v = img.ravel() / 255.0
        vectors.append(v)


    if num_to_keep > len(vectors):
        print("WARNING: you requested to keep more vectors that there are.  Reseting to the number of vectors (%i)" % len(vectors))
        num_to_keep = len(vectors)


    # write it out
#    out_fp = "data-mnist_%s_%i.tsv" % (dataset, num_to_keep)
    out_fp = "%s-data.tsv" % args.out_base
    with open(out_fp, "w") as out_file:
        out_file.write("#ignore this header\n")
        for vector in vectors[:num_to_keep]:
            out_file.write("\t".join(map(str, vector)) + "\n")

#    out_fp = "labels-mnist_%s_%i.txt" % (dataset, num_to_keep)
    out_fp = "%s-labels.txt" % args.out_base
    with open(out_fp, "w") as out_file:
        out_file.write("#ignore this header\n")
        for label in labels[:num_to_keep]:
            out_file.write("%s\n" % label)

    print("\nDone.")

    return


if __name__ == "__main__":
    main()
