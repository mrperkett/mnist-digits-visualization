# Background
TODO

![MNIST digits mapped to two dimensions using t-SNE](/images/MNIST-tSNE_mapping.png)



# Setup
## Install code
```
git clone https://github.com/mrperkett/mnist-digits-visualization mnist-digits-visualization
cd mnist-digits-visualization/
# TODO: add pyenv commands
sudo pip install -e .
```

## build bhtsne
```
git clone https://github.com/lvdmaaten/bhtsne.git bhtsne
cd bhtsne/
g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
cd ../
```

## install UMAP
TODO

## Download MNIST data
```
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
```
python scripts/prepareData.py --data data/ -t training -n 1000 -o input
python bhtsne/bhtsne.py -d 2 -p 30 -r 0 -i input-data.tsv -o tsne-out.tsv
```

# plot using jupyter lab
