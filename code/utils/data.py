import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from scipy.sparse import vstack

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle




def make_train_dataset(path, one_hot=False, chunk_size=50000):
	"""
	Builds the training dataset by reading an input CSV file.

	Arguments
	---------
	path : str
		Path to the input file.
		This file should be in CSV format, with the following columns:
		attribute_1, ..., attribute_m, label, num_occurences.
	one_hot : bool, default=False
		If true, the dataset is one-hot encoded and returned as a
		sparse matrix.
	chunk_size : int, default=50000
		Number of rows to one-hot encode and duplicate simultaneously
		(useful only when one_hot is true).

	Returns
	-------
	dataset : array of shape (n_instances, n_attributes + 1) or sparse
		matrix of shape (n_instances, n_one_hot_features)
		Training samples.
	y : int
		Target value (useful only for the PTF model).
	encoders : list or object
		If one_hot is False, this is a list of LabelEncoder instances,
		one for each attribute.
		Otherwise, it is a single OneHotEncoder instance.

	"""

	dat = pd.read_csv(path)
	X = dat.to_numpy()[:, :-2]
	counts = dat['count'].to_numpy()
	if one_hot:
		encoders = OneHotEncoder().fit(X)
		dataset = vstack([
			encoders.transform(
				np.concatenate([
					np.tile(X[j, :], (counts[j], 1))
					for j in range(i, min(i + chunk_size, X.shape[0]))
				])
			)
			for i in range(0, X.shape[0], chunk_size)
		])
		arities = np.array([
			cat.shape[0] for cat in encoders.categories_
		])
	else:
		m = X.shape[1]
		encoders = [
			LabelEncoder().fit(X[:, i])
			for i in range(m)
		]

		dataset = np.stack(
			[
				encoders[i].transform(X[:, i])
				for i in range(m)
			] + [counts],
			axis=1
		)
		arities = np.array([len(e.classes_) for e in encoders])
	dataset = shuffle(dataset, random_state=0)
	y = int(arities.prod()/X.shape[0])

	return dataset, y, encoders


def make_test_dataset(path, encoders, one_hot=False):
	"""
	Builds the test dataset by reading an input CSV file.

	Arguments
	---------
	path : str
		Path to the input file.
		This file should be in CSV format, with the following columns:
		attribute_1, ..., attribute_m, num_occurences,
		num_anomalous_occurrences.
	encoders : iterable of shape (n_attributes,) or object
		Categorical encoders returned by make_train_dataset when
		extracting the training set (list of LabelEncoder instances if
		one_hot is false, and OneHotEncoder instance otherwise).
	one_hot : bool, default=False
		If true, the dataset is one-hot encoded and returned as a
		sparse matrix.

	Returns
	-------
	dataset : list or tuple
		Test samples.
		If one_hot is false, this is a list of tuples containing a
		LongTensor representing the attributes and a LongTensor
		representing the label.
		Otherwise, this is a tuple containing a sparse matrix of shape
		(n_instances, n_one_hot_features) and an array of shape
		(n_instances,) representing the instances and their labels,
		respectively.

	"""

	dat = pd.read_csv(path)
	X = dat.to_numpy()
	if one_hot:
		classes_sets = [
			set(cat.tolist()) for cat in encoders.categories_
		]
	else:
		classes_sets = [set(e.classes_) for e in encoders]
	m = len(classes_sets)
	Xsub = np.stack(
		[
			x for x in X
			if sum(int(x[i] in classes_sets[i]) for i in range(m)) == m
		]
	)
	lab = Xsub[:, -1]

	if one_hot:
		Xenc = encoders.transform(Xsub[:, :-1])
		dataset = (Xenc, lab)
	else:
		dataset = []
		Xenc = np.stack(
			[
				encoders[i].transform(Xsub[:, i])
				for i in range(m)
			],
			axis=1
		)
		for i in range(Xenc.shape[0]):
			dataset.append((
				torch.from_numpy(Xenc[i, :].astype(np.int64)),
				torch.LongTensor([lab[i]])
			))

	return dataset


def duplicate_samples(dataset):
	X_0, counts = dataset[:, :-1], dataset[:, -1]
	X = np.concatenate([
		np.tile(x, (counts[i], 1))
		for i, x in enumerate(X_0)
	])
	return X