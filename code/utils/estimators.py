from abc import ABCMeta, abstractmethod

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.base import BaseEstimator




TEST_BATCH_SIZE = 5000


class CategoricalAD(BaseEstimator, metaclass=ABCMeta):
	"""
	Abstract parent class of the categorical anomaly detectors.

	"""

	@abstractmethod
	def fit(self, X, y=None):
		return self


	def score(self, X, y=None):
		"""
		Evaluate estimator on a validation set.

		Arguments
		---------
		X : array of shape (n_samples, n_attributes)
			Validation dataset.
			Lines whose $j$-th element is greater than self.arities[j]
			are filtered out.
		y : Ignored
			Not used, included for consistency with the scikit-learn
			API.

		Returns
		-------
		score : float
			Evaluation metric for the given validation set (the higher,
			the better).

		"""

		dataset = [
			torch.LongTensor(X[i, :len(self.arities)])
			for i in range(X.shape[0])
			if sum(
				int(X[i, j] < n) for j, n in enumerate(self.arities)
				) == len(self.arities)
		]

		dataloader = DataLoader(
			dataset,
			batch_size=TEST_BATCH_SIZE,
			shuffle=False,
			pin_memory=True
		)

		res = 0
		self.model.eval()
		with torch.no_grad():
			for inputs in dataloader:
				inputs = inputs.to(self.device)
				res -= self.model(inputs).sum().data.item()

		return res


	def score_samples(self, X):
		"""
		Computes the anomaly scores of the input samples.

		Arguments
		---------
		X : LongTensor of shape (n_samples, n_attributes)
			Input samples.
			Each element in the j-th column must be less than
			self.arities[j].

		Returns
		-------
		Y : array of shape (n_samples,)
			Anomaly scores of the input samples (the higher, the more
			anomalous).

		"""

		self.model.eval()
		X = X.to(self.device)
		return self.model(X).cpu().numpy()
