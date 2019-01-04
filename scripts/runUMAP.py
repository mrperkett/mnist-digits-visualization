#! /usr/bin/env python
"""
"""

import time

import numpy as np
import umap


def main():
    """
    """
    # user parameters
    inp_fp = "data-training-60k.tsv"
    out_fp = "out-training-60k.tsv"
#    inp_fp = "data-small.tsv"
#    out_fp = "out.tsv"

    # load data
    print("\nReading data...")
    data = []
    with open(inp_fp, "r") as inp_file:
        for line in inp_file:
            if line[0] == "#":
                continue
            data.append(map(float, line.rstrip("\n\r").split("\t")))
    data = np.array(data)

    # run UMAP
    print("\nRunning UMAP...")
    start_time = time.time()
    embedding = umap.UMAP(n_neighbors=10, min_dist=0.001, metric='correlation').fit_transform(data)
    dt = time.time() - start_time
    print("\ttime: %2.2f secs" % dt)


    # save
    print("\nWriting results to file...")
    with open(out_fp, "w") as out_file:
        for x, y in embedding:
            out_file.write("%f\t%f\n" % (x, y))


    print("\nDone.")

    return


if __name__ == "__main__":
    main()

