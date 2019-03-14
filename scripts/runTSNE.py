#! /usr/bin/env python3
"""
A script demonstrating how to call t-SNE on MNIST digits
"""

import argparse
import os
import sys
import time

import numpy as np
import sklearn.manifold

def parse_input(inp_fp):
    """
    """
    data = []
    prev_row_len = None
    with open(inp_fp, "r") as inp_file:
        for line in inp_file:
            if line.startswith("#"):
                continue

            row = list(map(float, line.rstrip("\n\r").split("\t")))

            if prev_row_len:
                assert len(row) == prev_row_len
            else:
                prev_row_len = len(row)

            data.append(row)

    return np.array(data)


def main():
    """
    Usage: ./runTSNE.py <inp_fp>

        inp_fp: input file path for raw data (tab-separated values; 1 vector per row)
    """
    # command line arguments
    if len(sys.argv) != 3:
        print(main.__doc__)
        sys.exit(1)

    inp_fp = sys.argv[1]
    out_fp = sys.argv[2]

    # user parameters
    perplexity = 30
    angle = 0.5
    metric = "euclidean"

    # make output directory if it doesn't exist
    out_dir = os.path.dirname(out_fp)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


    print("\nParsing input file...")
    data = parse_input(inp_fp)


    print("\nRunning t-SNE...")
    start_time = time.time()

    tsne = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12.0, 
                learning_rate=200.0, n_iter=5000, n_iter_without_progress=300, min_grad_norm=1e-07,
                metric=metric, init='pca', verbose=0, random_state=0, method="barnes_hut",
                angle=angle)
    tsne_projection = tsne.fit_transform(data)

    dt = time.time() - start_time
    print("\tTime: %f seconds" % dt)

    
   # write to file
#   out_fp = "%s-projection-angle=%f_perplexity=%f.tsv" % (out_base, angle, perplexity)
    with open(out_fp, "w") as out_file:
       out_file.write("#TSNE1\tTSNE2\n")
       for tsne1, tsne2 in tsne_projection:
           out_file.write("%f\t%f\n" % (tsne1, tsne2))


    print("\nDone.")

    return


if __name__ == "__main__":
    main()

