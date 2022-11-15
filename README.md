# Deep Categorical Anomaly Detection

This repository contains the code associated with our paper
"Deep Categorical Anomaly Detection", along with the data we used for
our experiments.

### Contents

There are two main directories:

* `code`: implementation of several categorical anomaly detection models,
  and Python script enabling their application to CSV-formatted datasets.
* `data`: preprocessed datasets used in our experiments.

### Setup and requirements

The code is written in Python 3.9.
To install the necessary dependencies:
```
pip install -r requirements.txt
```
This is sufficient for deep anomaly detection models, APE and Isolation
Forest to run.
In addition, the [PyCP_APR](https://lanl.github.io/pyCP_APR/) package
is required in order to use Poisson Tensor Factorization (PTF).

### Usage

Model training and evaluation can be performed using
```run_experiment.py```.
To display the available options, use
```
$ python run_experiment.py --help
```
The script ```experiments.sh``` reproduces all the experiments presented
in our paper.
Note that this is a very compute-intensive process.