import numpy as np

import torch

from sklearn.base import BaseEstimator

from pyCP_APR import CP_APR




class PTF(BaseEstimator):
	"""
	Scikit-learn wrapper for the pyCP_APR implementation of Poisson
	Tensor Factorization [1,2].


	Parameters
	----------
	K : int, default=128
		Dimension :math:`d` of the attribute embeddings.
	n_iters : int, n_iters=1000
		Number of training iterations.
	random_state : int, default=None
		Seed of the random number generator.

	Attributes
	----------
	model : object
		Instance of the CP_APR class.
		Implements the PTF model.

	References
	----------
	[1] https://lanl.github.io/pyCP_APR/
	[2] Eren, Maksim, et al. Multi-Dimensional Anomalous Entity
		Detection via Poisson Tensor Factorization. ISI, 2020.

	"""

	def __init__(
		self,
		K=128,
		n_iters=1000,
		random_state=None
		):

		self.model = None
		self.K = K
		self.n_iters=n_iters
		self.random_state=random_state

	def fit(self, X, y):
		"""
		Fit estimator.

		Arguments
		---------
		X : array of shape (n_samples, n_attributes)
			Training dataset.
		y : array of shape (n_samples,)
			Target values.

		Returns
		-------
		self : object
			Fitted estimator.

		"""

		self.model = CP_APR(
			method='torch',
			device='gpu' if torch.cuda.is_available() else 'cpu',
			return_type='numpy',
			n_iters=self.n_iters,
			verbose=0,
			dtype='torch.FloatTensor',
			random_state=self.random_state
		)
		self.model.fit(
			coords=X,
			values=y,
			rank=[1, self.K]
		)
		self.losses = list(
			self.model.get_params()['logLikelihoods']
		)

		return self

	def score_samples(self, X):
		"""
		Computes the anomaly scores of the input samples.

		Arguments
		---------
		X : array of shape (n_samples, n_attributes)
			Input samples.

		Returns
		-------
		Y : array of shape (n_samples,)
			Anomaly scores of the input samples (the higher, the more
			anomalous).

		"""

		return 1-self.model.predict_scores(
			coords=X,
			values=np.ones(X.shape[0])
		)

	def score(self, X, y):
		"""
		Evaluate estimator on a validation set.

		Arguments
		---------
		X : array of shape (n_samples, n_attributes)
			Validation dataset.
		y : array of shape (n_samples,)
			Target values.

		Returns
		-------
		score : float
			Evaluation metric for the given validation set (the higher,
			the better).

		"""

		cmax = [
			self.model.M[0]['Factors'][str(i)].shape[0]
			for i in range(X.shape[1])
		]
		ind = [
			i for i in range(X.shape[0])
			if sum(int(X[i,j]<m) for j, m in enumerate(cmax)) == len(cmax)
		]
		return self.model.predict_scores(
			coords=X[ind,:],
			values=y[ind]
		).sum()
