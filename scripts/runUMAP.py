#! /usr/bin/env python
"""
"""

import os
import sys
import time

import numpy as np
import umap

sys.path.append(os.path.abspath(os.path.join((os.path.dirname(__file__)), "..")))
from utils.mnist import read_mnist_data


def main():
    """
    """
    # user parameters
    mnist_data_folder = "data"
    out_base = "test"
#    inp_fp = "data-training-60k.tsv"
#    out_fp = "out-training-60k.tsv"
#    inp_fp = "data-small.tsv"
#    out_fp = "out.tsv"

    # load data
    print("\nReading data...")
    labels, images, data = read_mnist_data(dataset="training", path=mnist_data_folder, 
            num_to_keep=10000)

    # run UMAP
    print("\nRunning UMAP...")
    start_time = time.time()
    embedding = umap.UMAP(n_neighbors=10, min_dist=0.001, metric='correlation').fit_transform(data)
    dt = time.time() - start_time
    print("\ttime: %2.2f secs" % dt)


    print("\nWriting results to file...")
    # UMAP projection
    out_fp = "%s-umap_projection.tsv" % out_base
    with open(out_fp, "w") as out_file:
        for x, y in embedding:
            out_file.write("%f\t%f\n" % (x, y))

    # labels
    out_fp = "%s-labels.txt" % out_base
    with open(out_fp, "w") as out_file:
        for label in labels:
            out_file.write("%s\n" % label)


    print("\nDone.")

    return


if __name__ == "__main__":
    main()

