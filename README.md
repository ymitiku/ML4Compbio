# Application of Machine Learning to Life Science(Biology)
This repository contains code for blog post [here](https://medium.com/@se.mitiku.yohannes/application-of-machine-learning-to-life-science-biology-part-1-76924b6c3fa3#03b2-b881970a4ea).
## Part 1 Measuring similarity of Animal Genomes
In this tutorial we have seen that how to use kmer features to compute similarity between two genome sequences. Here I will provide how to use the scripts to generate embedding that could be visualized using tensorboard. Beside the generation of embedding we can also use the script to find the most similar animal to the given animal

## Requirements
* tensorflow 2.x
* numpy 
* torch
* tensorboard

## Downloading the dataset
Use the following command to download the sample dataset
```bash
curl -L -o data/sb008.fastz "https://drive.google.com/uc?export=download&id=1mJpltSs1negIBkzFSWYHcF8MyvnnAbrx"
```

## How to generate embeddings and visualize using tensorboard
The following bash script can be used to generate the embeddings for each animal genome in the given file.
```bash
python -m scripts.part1 -p data/sb008.fastz -t embd
```
After the above scipt run, use the following script to view the embeddings on tensorboad
```bash
tensorboard --logdir=runs
```
## How to find most similar animal to a given animal
The same script can be used to view most similar animal for certain animal
```bash
python -m scripts.part1 -p data/sb008.fastz -t sim -a rat -k 5
```