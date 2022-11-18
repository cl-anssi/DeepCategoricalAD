# Datasets

This directory contains the two preprocessed datasets used in our
experiments.
Each dataset consists of two CSV files (training and test sets), each line
of which represents one distinct instance.
An additional field in the training set (`count`) indicates the number of
occurrences of each instance.

A few preprocessing steps were applied to the original datasets: for
categorical attributes, we replaced rare values with a single "OTHER" token,
which is also used for values appearing in the test set but not in the
training set.
As for numerical features, we turned them to categorical attributes by
applying k-means clustering (with k=20) to each feature separately.
Note that the number of occurrences of categorical values (used to define rare
values) and the clusters of numerical values were both computed on the
training set only in order to avoid data snooping.

### UNSW-NB15 dataset

The original data can be found
[here](https://research.unsw.edu.au/projects/unsw-nb15-dataset), along with
links to the papers describing the dataset.

### IDS2017 dataset

The original data can be found
[here](https://www.unb.ca/cic/datasets/ids-2017.html), along with a
description of the simulated infrastructure and a timeline of attacks.
