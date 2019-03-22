#! /usr/bin/env python
"""
An example script demonstrating how to run t-SNE using sklearn on the MNIST data set.
"""

import argparse
import os
import sys
import time

import numpy as np
import sklearn.manifold

sys.path.append(os.path.abspath(os.path.join((os.path.dirname(__file__)), "..")))
from utils.mnist import read_mnist_data


def parse_args(arg_list):
    """
    Parse command line arguments using argparse
    """
    parser = argparse.ArgumentParser()

    # user arguments
    parser.add_argument("--data-dir", "-d", dest="mnist_data_dir", type=str, required=True,
            help="Directory containing the downloaded MNIST digits data.")
    parser.add_argument("--data-set", "-s", dest="mnist_data_set", type=str, required=False, 
            default="all", help="Data type: 'training', 'test', or 'all'")
    parser.add_argument("-n", dest="num_to_keep", type=int, default=None, required=False,
            help="Read the first n digits.  Default: read all digits for the given data type.")
    parser.add_argument("--out-base", "-o", dest="out_base", type=str, required=True,
            help="The output file base: 1) <out_base>-tsne_projection.tsv 2) <out_base>-tsne_labels.txt")

    # arguments passed through to tSNE
    parser.add_argument("--perplexity", dest="perplexity", type=int, required=False,
            default=30, help="tSNE perplexity")
    parser.add_argument("--angle", dest="angle", type=float, required=False,
            default=0.5, help="tSNE angle")
    parser.add_argument("--metric", dest="metric", type=str, required=False,
            default="euclidean", help="tSNE metric")

    args = parser.parse_args(arg_list)
    

    return args


def main():
    """
    """
    args = parse_args(sys.argv[1:])

    # make output directory if it doesn't exist
    out_dir = os.path.dirname(args.out_base)
    if out_dir != "" and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    print("\nReading data...")
    labels, images, data = read_mnist_data(dataset=args.mnist_data_set, path=args.mnist_data_dir, 
            num_to_keep=args.num_to_keep)


    print("\nRunning t-SNE...")
    start_time = time.time()

    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=args.perplexity, early_exaggeration=12.0,
                learning_rate=200.0, n_iter=5000, n_iter_without_progress=300, min_grad_norm=1e-07,
                metric=args.metric, init='pca', verbose=0, random_state=0, method="barnes_hut",
                angle=args.angle)
    tsne_projection = tsne.fit_transform(data)

    dt = time.time() - start_time
    print("\tTime: %f seconds" % dt)


    print("\nWriting results to file...")
    # projection
    out_fp = "%s-tsne_projection.tsv" % args.out_base
    with open(out_fp, "w") as out_file:
        out_file.write("#TSNE1\tTSNE2\n")
        for x, y in tsne_projection:
            out_file.write("%f\t%f\n" % (x, y))

    # labels
    out_fp = "%s-tsne_labels.txt" % args.out_base
    with open(out_fp, "w") as out_file:
        for label in labels:
            out_file.write("%s\n" % label)


    print("\nDone.")

    return


if __name__ == "__main__":
    main()

