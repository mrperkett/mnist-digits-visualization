"""
I have modified from the following:
https://gist.github.com/akesling/5358964

Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

import os
import struct
import numpy as np


def read_mnist_data(dataset="training", path=".", num_to_keep=None):
    """
    Read the MNIST data set and return it in a useful form.

    Args:
        dataset (str): which dataset to load "training", "test", "all"
        path (str): path to directory containing the following MNIST handwritten digit files
                "t10k-images-idx3-ubyte"
                "t10k-labels-idx1-ubyte"
                "train-images-idx3-ubyte"
                "train-labels-idx1-ubyte"
        num_to_keep (int): number of digits to keep (a value of None means to keep all)

    Return:
        labels (list<str>)
        images (np.array)
        vectors (np.array)
    """
    # TODO:     1) modify this to allow for gzipped files
    #           2) this dataset is small, but memory isn't used efficiently

    # which files to load
    image_fps, label_fps = [], []
    if dataset in ("training", "all"):
        image_fps.append(os.path.join(path, 'train-images-idx3-ubyte'))
        label_fps.append(os.path.join(path, 'train-labels-idx1-ubyte'))

    if dataset == ("testing", "all"):
        image_fps.append(os.path.join(path, 't10k-images-idx3-ubyte'))
        label_fps.append(os.path.join(path, 't10k-labels-idx1-ubyte'))

    if len(image_fps) == 0:
        raise ValueError("dataset must be 'testing', 'training', or 'all'")

    # Load everything into numpy arrays
    for label_fp, image_fp in zip(label_fps, image_fps):
        labels = None
        images = None

        # labels file
        with open(label_fp, 'rb') as inp_file:
            magic, num = struct.unpack(">II", inp_file.read(8))
            t = np.fromfile(inp_file, dtype=np.int8)
            num_labels = len(t)
            
            # concatenate to growing list
            if labels:
                np.concatenate(labels, t, axis=0)
            else:
                labels = t

        # images file
        with open(image_fp, 'rb') as inp_file:
            magic, num, rows, cols = struct.unpack(">IIII", inp_file.read(16))
            t = np.fromfile(inp_file, dtype=np.uint8).reshape(num_labels, rows, cols)
            
            # concatenate to growing list
            if images:
                np.concatenate(images, t, axis=0)
            else:
                images = t

    # build vectors
    # each 28 x 28 array is transformed to a length 784 vector and normalized between 0.0 and 1.0
    vectors = np.full((len(labels), 784), 0.0, dtype=np.double)
    for i in range(len(images)):
        vectors[i] = images[i].ravel() / 255.0
   

    # resize each to return up to num_to_keep
    assert len(labels) == len(images) == len(vectors)
    if num_to_keep is not None:
        if len(labels) < num_to_keep:
            raise AssertionError("Requested number of values to keep (%i) is greater than the total number of values (%i)" % (num_to_keep, len(labels)))

        resize_shape = list(labels.shape)
        resize_shape[0] = num_to_keep
        labels = np.resize(labels, resize_shape)

        resize_shape = list(images.shape)
        resize_shape[0] = num_to_keep
        images = np.resize(images, resize_shape)

        resize_shape = list(vectors.shape)
        resize_shape[0] = num_to_keep
        vectors = np.resize(vectors, resize_shape)

    return labels, images, vectors


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

    return
