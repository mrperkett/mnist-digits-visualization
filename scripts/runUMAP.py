#! /usr/bin/env python
"""
An example script demonstrating how to run UMAP on the MNIST data set.
"""

import argparse
import os
import sys
import time

import numpy as np
import umap

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
            help="The output file base: 1) <out_base>-umap_projection.tsv 2) <out_base>-labels.txt")

    # arguments passed through to UMAP
    parser.add_argument("--n_neighbors", dest="n_neighbors", type=int, required=False,
            default=10, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", dest="min_dist", type=float, required=False,
            default=0.001, help="UMAP min_dist")
    parser.add_argument("--metric", dest="metric", type=str, required=False,
            default="correlation", help="UMAP metric")

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

    print("\nRunning UMAP...")
    start_time = time.time()
    embedding = umap.UMAP(  n_neighbors=args.n_neighbors,
                            min_dist=args.min_dist, 
                            metric=args.metric ).fit_transform(data)
    dt = time.time() - start_time
    print("\ttime: %2.2f secs" % dt)


    print("\nWriting results to file...")
    # projection
    out_fp = "%s-umap_projection.tsv" % args.out_base
    with open(out_fp, "w") as out_file:
        out_file.write("#UMAP1\tUMAP2\n")
        for x, y in embedding:
            out_file.write("%f\t%f\n" % (x, y))

    # labels
    out_fp = "%s-umap_labels.txt" % args.out_base
    with open(out_fp, "w") as out_file:
        for label in labels:
            out_file.write("%s\n" % label)

    # images
    out_fp = "%s-umap_images.npy" % args.out_base
    np.save(out_fp, images)

    print("\nDone.")

    return


if __name__ == "__main__":
    main()

