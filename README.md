# Background
t-SNE and UMAP are popular dimensional reduction algorithms that allow you to visualize high dimensional data in two dimensions.  This repository contains my investigation of these techniques on the popular MNIST handwritten digits dataset.  It includes the code necessary to generate all the plots you see here.

## Gallery
The MNIST dataset contains 70,000 handwritten digits, which have been scanned and resized to 28x28 pixel images.  A random subset of the handwritten digits is provided below. 

Each image can be represented as a vector of length 28 x 28 = 784, where the pixel values correspond to the level of color for each pixel (0.0 for white to 1.0 for black).  Using dimensional reduction algorithms, we can map each vector from 784 dimensions down to two dimensions for visualization.

To illustrate the power of t-SNE and UMAP, it's useful to see what you get using another powerful technique, Principal Component Analysis (PCA).  Although there is some separation between images in the plot below, note that the clusters would be impossible to distinguish if the data weren't labeled (i.e. colored).

![MNIST digits mapped to two dimensions using t-SNE](/images/MNIST-PCA_mapping_color.png)


However, if we use t-SNE, we can immediately see beautiful organization!
![MNIST digits mapped to two dimensions using t-SNE](/images/MNIST-tSNE_mapping.png)

And similar for UMAP
![MNIST digits mapped to two dimensions using UMAP](/images/MNIST-UMAP_mapping.png)

And by replacing the colored points from the t-SNE plot with their corresponding handwritten digit, we can zoom in and view the striking local organization.  In the plot below, notice the smooth transition from open top 4's to closed top 4's to 9's.  Points near each other are visually more similar than points distant.
![MNIST digits mapped to two dimensions using UMAP](/images/MNIST-tSNE-zoom_in_fours_and_nines.png)

t-SNE, UMAP, and other dimensional reduction techniques are useful in a variety of fields including bioinformatics, where we use them to  visualize the results from single cell RNA-Seq experiments.  Each cell is a high dimensional data point with the expression level of each gene as its coordinates.

## References
TODO: add the best of the references that I've found


# Setup
## Installing
Clone the repository.
```
git clone https://github.com/mrperkett/mnist-digits-visualization mnist-digits-visualization
cd mnist-digits-visualization/
```

Install requirements in a python virtual environment.
```
python3 -m venv testenv
. testenv/bin/activate
python3 -m pip install -r requirements.txt
```

## build bhtsne
```
git clone https://github.com/lvdmaaten/bhtsne.git bhtsne
cd bhtsne/
g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
cd ../
```

## Download MNIST data
```
cd ../
mkdir data
cd data/

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gzip -d *.gz
cd ../
```

# Running t-SNE

# Running UMAP
```
usage: runUMAP.py [-h] --data-dir MNIST_DATA_DIR [--data-set MNIST_DATA_SET]
                  [-n NUM_TO_KEEP] --out-base OUT_BASE
                  [--n_neighbors N_NEIGHBORS] [--min_dist MIN_DIST]
                  [--metric METRIC]

optional arguments:
  -h, --help            show this help message and exit
  --data-dir MNIST_DATA_DIR, -d MNIST_DATA_DIR
                        Directory containing the downloaded MNIST digits data.
  --data-set MNIST_DATA_SET, -s MNIST_DATA_SET
                        Data type: 'training', 'test', or 'all'
  -n NUM_TO_KEEP        Read the first n digits. Default: read all digits for
                        the given data type.
  --out-base OUT_BASE, -o OUT_BASE
                        The output file base: 1)
                        <out_base>-umap_projection.tsv 2)
                        <out_base>-labels.txt
  --n_neighbors N_NEIGHBORS
                        UMAP n_neighbors
  --min_dist MIN_DIST   UMAP min_dist
  --metric METRIC       UMAP metric
```

Example call on small subset of MNIST dataset
```
python3 ./mnist-digits-visualization/scripts/runUMAP.py -d data/ -s all -n 5000 -o test
```

If all goes well, you should see output similar to that below, and two output files will be generated (`example-umap_projection.tsv` and `test-labels.txt`).  Example Jupyter notebooks are provided in the `notesbooks/` folder that can be used to generate plots showing the UMAP projection.
```
Reading data...

Running UMAP...
        time: 20.09 secs

Writing results to file...

Done.
```

This script allows you to set three common UMAP parameters from the command line (`n_neighbors`, `min_dist`, and `metric`).  You can see all the UMAP parameters available in the documentation (https://umap-learn.readthedocs.io/en/latest/).


# plot using jupyter lab
