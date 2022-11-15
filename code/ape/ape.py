import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch import optim

from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler, autocast

from utils.data import duplicate_samples
from utils.estimators import CategoricalAD
from utils.modules import PairwiseInteractions




class ApeModule(nn.Module):
	"""
	Implementation of the APE model, after [1].


	Parameters
	----------
	arities : iterable of shape (n_attributes,)
		Interable of integers representing the number of possible
		values for each categorical attribute.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	n_noise_samples : int, default=3
		Number of negative samples per attribute.
		For each positive sample in a minibatch,
		(n_noise_samples * n_attributes) negative samples are drawn.
	noise_dist : iterable of shape (n_attributes,), default=None
		Iterable of Tensors representing the (possibly unnormalized)
		noise distribution for each attribute.
		The :math:`j`-th Tensor thus should have shape (arities[j],).
		If None, a uniform noise distribution is used.

	Attributes
	----------
	m : int
		Number of attributes.
	embeddings : nn.ModuleList of shape (n_attributes,)
		Embeddings of the attribute values.
		The :math:`j`-th element of the list is an embedding table of
		shape (arities[j], embedding_dim).
	pi : nn.Module
		PyTorch module computing all pairwise products between the
		attribute embeddings.
	weights : Tensor of shape (m, m)
		Learnable weights associated with the attribute pairs.
	c : Tensor of shape (1,)
		Learnable estimate of the log-partition function.
	bce : nn.Module
		PyTorch module computing the binary cross-entropy loss.
	training : bool
		Controls the behavior of the forward() method, which is
		different between training and inference.

	References
	----------
	[1] Chen, Ting, et al. "Entity embedding-based anomaly detection
		for heterogeneous categorical events". IJCAI, 2016.

	"""

	def __init__(
		self,
		arities,
		embedding_dim=128,
		n_noise_samples=3,
		noise_dist=None
		):

		super(ApeModule, self).__init__()

		self.m = len(arities)

		self.embeddings = nn.ModuleList(
			nn.Embedding(n, embedding_dim)
			for n in arities
		)

		self.pi = PairwiseInteractions(self.m)

		self.weights = nn.Parameter(
			torch.randn(self.m * (self.m - 1) // 2)
		)
		self.c = nn.Parameter(torch.zeros(1))

		if noise_dist is not None:
			self.noise_dist = [ndist/ndist.sum() for ndist in noise_dist]
		else:
			self.noise_dist = [torch.ones(n)/n for n in arities]

		self.n_noise_samples = n_noise_samples

		self.bce = nn.BCEWithLogitsLoss()

		self.training = True


	def forward(self, inputs):
		"""
		Forward pass.

		Arguments
		---------
		inputs : LongTensor of shape (batch_size, n_attributes)
			Input samples.

		Returns
		-------
		y : Tensor
			If the model is in training mode, y has shape (1,) and
			contains the total loss for the input samples.
			Otherwise, y has shape (batch_size,) and contains the
			anomaly scores of the input samples.

		"""

		if self.noise_dist[0].device != inputs.device:
			for i in range(self.m):
				self.noise_dist[i] = self.noise_dist[i].to(inputs.device)
		if self.training:
			return self._loss(inputs)
		else:
			return -self._log_probability(inputs)


	def _get_neg_samples(self, inputs):
		"""
		Draws negative samples for the given minibatch.

		Arguments
		---------
		inputs : LongTensor of shape (batch_size, n_attributes)
			Input samples.

		Returns
		-------
		neg : LongTensor of shape
			(batch_size * n_attributes * n_noise_samples, n_attributes)
			Negative samples.
		probs : Tensor of shape
			(batch_size * n_attributes * n_noise_samples,)
			Probabilities of the negative samples under the noise
			distribution.

		"""

		n = inputs.shape[0]
		neg = inputs.repeat((self.m*self.n_noise_samples, 1))
		idx = torch.cat([
			neg.new_tensor(
				[i]*(self.n_noise_samples*n),
				dtype=torch.long
			) for i in range(self.m)
		])
		probs = neg.new_ones(neg.shape[0], dtype=torch.float)
		for i in range(self.m):
			k, l = n*self.n_noise_samples*i, n*self.n_noise_samples*(i+1)
			neg[k:l, i] = torch.multinomial(
				self.noise_dist[i],
				self.n_noise_samples*n,
				replacement=True
			)
			probs[k:l] = self.noise_dist[i][neg[k:l, i]]

		return neg, probs


	def _log_probability(self, inputs):
		"""
		Computes the estimated log-probability of the given samples.

		Arguments
		---------
		inputs : LongTensor of shape (batch_size, n_attributes)
			Input samples.

		Returns
		-------
		probs : Tensor of shape (batch_size,)
			Estimated log-probabilities of the input samples.

		"""

		X = torch.stack([
			self.embeddings[i](inputs[:, i])
			for i in range(self.m)
		], dim=1)
		y = self.pi(X).sum(2).matmul(self.weights)
		return y + self.c


	def _loss(self, inputs):
		"""
		Computes the NCE loss for the given samples.

		Arguments
		---------
		inputs : LongTensor of shape (batch_size, n_attributes)
			Input samples.

		Returns
		-------
		loss : Tensor of shape (1,)
			NCE loss for the minibatch.

		"""

		neg, probs = self._get_neg_samples(inputs)
		preds = self._log_probability(
			torch.cat([
				neg, inputs
			])
		)
		preds -= torch.cat([
			torch.log(probs),
			torch.log(
				torch.stack([
					self.noise_dist[i][inputs[:, i]]
					for i in range(self.m)
				], dim=1)
			).mean(1)
		])
		labels = torch.cat([
			preds.new_zeros(neg.shape[0]),
			preds.new_ones(inputs.shape[0])
		])
		return self.bce(preds, labels)


class APE(CategoricalAD):
	"""
	Scikit-learn wrapper for the APE module.


	Parameters
	----------
	arities : iterable of shape (n_attributes,)
		Interable of integers representing the number of possible
		values for each categorical attribute.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	n_noise_samples : int, default=3
		Number of negative samples per attribute.
		For each positive sample in a minibatch,
		(n_noise_samples * n_attributes) negative samples are drawn.
	batch_size : int, default=128
		Minibatch size to use for training.
	epochs : int, default=20
		Number of training epochs to perform.
	lr : float, default=1e-3
		Learning rate of the Adam optimizer.
	weight_decay : float, default=1e-5
		Weight decay coefficient (L2 regularization) of the Adam
		optimizer.
	device : object, default=torch.device('cuda')
		PyTorch device where the model resides.

	Attributes
	----------
	model : nn.Module
		Instance of the ApeModule class.

	"""

	def __init__(
		self,
		arities,
		embedding_dim=128,
		n_noise_samples=3,
		batch_size=128,
		epochs=20,
		lr=1e-3,
		weight_decay=1e-5,
		device=torch.device('cuda')
		):

		self.arities = arities
		self.embedding_dim = embedding_dim
		self.n_noise_samples = n_noise_samples
		self.batch_size = batch_size
		self.epochs = epochs
		self.lr = lr
		self.weight_decay = weight_decay
		self.device = device

		self.model = None


	def fit(self, X, y=None):
		"""
		Fit estimator.

		Arguments
		---------
		X : array of shape (n_samples, n_attributes + 1)
			Training dataset.
			Each element in the j-th column must be less than
			self.arities[j].
			The last column contains the number of occurrences of each
			sample.
		y : Ignored
			Not used, included for consistency with the scikit-learn
			API.

		Returns
		-------
		self : object
			Fitted estimator.

		"""

		X_inflated = duplicate_samples(X)
		dataset = [
			torch.LongTensor(X_inflated[i, :])
			for i in range(X_inflated.shape[0])
		]

		ndist = []
		for i in range(len(self.arities)):
			idx, cnt = np.unique(
				X_inflated[:, i],
				return_counts=True
			)
			nd = torch.zeros(self.arities[i]).to(self.device)
			for j, n in zip(idx, cnt):
				nd[j] = n
			nd /= nd.sum()
			nd.clamp(min=1e-10)
			ndist.append(nd/nd.sum())

		self.model = ApeModule(
			self.arities,
			embedding_dim=self.embedding_dim,
			n_noise_samples=self.n_noise_samples
		).to(self.device)

		dataloader = DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=True,
			pin_memory=True
		)

		optimizer = optim.Adam(
			self.model.parameters(),
			lr=self.lr,
			weight_decay=self.weight_decay
		)
		scaler = GradScaler()

		self.model.training = True
		self.losses = []
		for epoch in range(self.epochs):
			running_loss = 0
			for inputs in dataloader:
				inputs = inputs.to(self.device)
				with autocast():
					loss = self.model(inputs)
				running_loss += loss.data.item()
				optimizer.zero_grad()
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			self.losses.append(running_loss)

		return self