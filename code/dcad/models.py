import math

from abc import ABCMeta

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from dcad.networks import make_network
from dcad.regularizers import *




class DeepCatAD(nn.Module, metaclass=ABCMeta):
	"""
	Abstract parent class of the deep categorical anomaly detection
	framework.


	Parameters
	----------
	arities : iterable of shape (n_attributes,)
		Interable of integers representing the number of possible
		values for each categorical attribute.
	architecture : str, default='dnn'
		Name of the neural architecture to use.
		Possible values: 'autoint', 'dnn', 'nfm'.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	hidden_dim : int, default=16
		Dimension :math:`d'` of the hidden representation space.
	hidden_layers : int, default=2
		Number of hidden layers of the neural architecture.
	n_heads : int, default=8
		Number of attention heads in the Multihead Attention blocks.
		Used for AutoInt only.
	dropout : float, default=0
		Dropout rate applied at each fully connected layer during
		training.
		Used for DNN and NFM only.
	residual_connections : bool, default=True
		Whether to add skip connections in fully connected networks.
		Used for DNN and NFM only.
	bias : bool, default=True
		Whether to include bias terms in linear layers.

	Attributes
	----------
	m : int
		Number of attributes.
	embeddings : nn.ModuleList of shape (n_attributes,)
		Embeddings of the attribute values.
		The :math:`j`-th element of the list is an embedding table of
		shape (arities[j], embedding_dim).
	nn : nn.Module
		Module implementing the neural architecture.
	training : bool
		Controls the behavior of the forward() method, which can
		differ between training and inference.

	"""

	def __init__(
		self,
		arities,
		architecture='dnn',
		embedding_dim=128,
		hidden_dim=16,
		hidden_layers=2,
		n_heads=8,
		dropout=0,
		residual_connections=True,
		bias=True
		):

		super(DeepCatAD, self).__init__()

		self.arities = arities
		self.m = len(arities)
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		self.embeddings = nn.ModuleList([
			nn.Embedding(a, embedding_dim)
			for a in arities
		])

		self.nn = make_network(
			architecture,
			n_inputs=self.m,
			embedding_dim=self.embedding_dim,
			hidden_dim=self.hidden_dim,
			hidden_layers=hidden_layers,
			dropout=dropout,
			residual_connections=residual_connections,
			bias=bias,
			n_heads=n_heads
		)

		self.training = True


	def forward(self, inputs, **kwargs):
		"""
		Forward pass of the model.

		Arguments
		---------
		inputs : LongTensor of shape (batch_size, n_attributes)
			Input samples.

		Other arguments
		---------------
		**kwargs : optional architecture-specific keyword arguments.

		Returns
		-------
		H : Tensor of shape (batch_size, hidden_dim)
			Hidden representations of the input samples.

		"""

		H = self._hidden_embedding(inputs)
		return H


	def _hidden_embedding(self, inputs):
		"""
		Computes the hidden representation of the input samples.

		Arguments
		---------
		inputs : LongTensor of shape (batch_size, n_attributes)
			Input samples.

		Returns
		-------
		H : Tensor of shape (batch_size, hidden_dim)
			Hidden representations of the input samples.

		"""

		X = torch.stack(
			[self.embeddings[i](inputs[:,i]) for i in range(self.m)],
			dim=1
		)
		return self.nn(X)


class DeepCatSVDD(DeepCatAD):
	"""
	Implementation of the DeepSVDD algorithm, after [1].
	This algorithm trains a neural network so that it maps the
	training data into a small hypersphere, which is learned jointly
	with the network parameters.
	Given the trained model, the anomaly score of a new sample is
	then the distance between its hidden representation and the center
	of the hypersphere.


	Parameters
	----------
	arities : iterable of shape (n_attributes,), default=None
		Interable of integers representing the number of possible
		values for each categorical attribute.
	architecture : str, default='dnn'
		Name of the neural architecture to use.
		Possible values: 'autoint', 'dnn', 'nfm'.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	hidden_dim : int, default=16
		Dimension :math:`d'` of the hidden representation space.
	hidden_layers : int, default=2
		Number of hidden layers of the neural architecture.
	n_heads : int, default=8
		Number of attention heads in the Multihead Attention blocks.
		Used for AutoInt only.
	dropout : float, default=0
		Dropout rate applied at each fully connected layer during
		training.
		Used for DNN and NFM only.
	residual_connections : bool, default=True
		Whether to add skip connections in fully connected networks.
		Used for DNN and NFM only.
	bias : bool, default=True
		Whether to include bias terms in linear layers.
	nu : float, default=.01
		Soft boundary hyperparameter.
		Controls the fraction of the training samples that fall outside
		of the data-enclosing hypersphere.
	reg : str, default=None
		Type of regularization to use (in addition to weight decay
		and dropout).
		See dcad.regularizers for more information on the available
		methods, and [2] for a detailed description of the
		regularization procedure.
		Possible values: 'noise', 'variance'.
	alpha : float, default=.9
		Controls how quickly the relative weight of the additional
		regularization term in the overall loss changes.
		With alpha=1, the weight is constant.
		With alpha=0, the weight is entirely recomputed at each
		minibatch, based on the ratio between the SVDD loss and the
		regularization term for the current minibatch.
		Used only when reg is not None.
	beta : float, default=.5
		Controls the ratio between the SVDD loss and the additional
		regularization term.
		Used only when reg is not None.
	n_noise_tasks : int, default=10
		Number of noisy classification tasks used to compute the
		noise regularization term.
		Used only when reg=='noise'.
	variance_threshold : float, default=.1
		When using variance-based regularization, the additional
		regularization term is added only when the total sample
		variance of the hidden features for the current minibatch
		falls below this threshold.
		Used only when reg=='variance'.
	annealing_rate : int, default=3
		Number of epochs between two decreases of the variance
		threshold.
		Used only when reg=='variance'.

	Attributes
	----------
	R : float
		Radius of the data-enclosing hypersphere.
	c : Tensor of shape (hidden_dim,)
		Center of the data-enclosing hypersphere.
	regularizer : nn.Module
		PyTorch module implementing the additional regularizer.
	reg_coeff : float
		Current weight of the additional regularization term.

	References
	----------
	[1] Ruff, Lukas, et al. "Deep one-class classification".
		ICML, 2018.
	[2] Chong, Penny, et al. "Simple and effective prevention of mode
		collapse in deep one-class classification". IJCNN, 2020.

	"""

	def __init__(
		self,
		arities=None,
		architecture='dnn',
		embedding_dim=128,
		hidden_dim=16,
		hidden_layers=2,
		n_heads=8,
		dropout=0,
		residual_connections=True,
		bias=True,
		nu=.01,
		reg=None,
		alpha=.9,
		beta=.5,
		num_noise_tasks=10,
		variance_threshold=.1,
		annealing_rate=3,
		**kwargs
		):

		super(DeepCatSVDD, self).__init__(
			arities,
			embedding_dim=embedding_dim,
			hidden_dim=hidden_dim,
			architecture=architecture,
			hidden_layers=hidden_layers,
			n_heads=n_heads,
			dropout=dropout,
			residual_connections=residual_connections,
			bias=bias
		)

		self.nu = nu
		self.R = 0
		self.c = nn.Parameter(
			torch.zeros(hidden_dim)
		)
		self.c.requires_grad = False

		self.reg = reg
		if reg == 'noise':
			self.regularizer = NoiseRegularizer(
				hidden_dim,
				num_noise_tasks
			)
		elif reg == 'variance':
			self.regularizer = VarianceRegularizer(
				variance_threshold,
				annealing_rate
			)

		self.alpha = alpha
		self.beta = beta
		self.reg_coeff = 0


	def forward(self, inputs, update_R=False, **kwargs):
		"""
		Forward pass of the model.

		Arguments
		---------
		inputs : LongTensor of shape (batch_size, n_attributes)
			Input samples.
		update_R : bool, default=False
			Whether to update the radius of the hypersphere.

		Other arguments
		---------------
		**kwargs : not used.

		Returns
		-------
		y : Tensor
			If the model is in training mode, y has shape (1,) and
			contains the total loss for the input samples.
			Otherwise, y has shape (batch_size,) and contains the
			anomaly scores of the input samples.

		"""

		H = self._hidden_embedding(inputs)
		dist = torch.pow(H-self.c.unsqueeze(0), 2).sum(1)

		if self.training:
			if update_R:
				self.R = np.quantile(
					np.sqrt(dist.clone().data.cpu().numpy()),
					1 - self.nu
				)
			hinge_loss = torch.fmax(
				dist - self.R**2,
				torch.zeros_like(dist)
			)
			loss = self.R**2 + hinge_loss.mean() / self.nu
			if self.reg is not None:
				reg_loss = self.regularizer(H)
				self._update_reg_coeff(
					loss.clone().detach(),
					reg_loss.clone().detach()
				)
				loss += self.reg_coeff * reg_loss
			return loss
		else:
			return dist - self.R**2


	def set_center(self, dataloader):
		"""
		Computes the center of the hypersphere as the mean hidden
		representation of a sequence of samples, then stores the
		result in self.c.

		Arguments
		---------
		dataloader : iterable
			Iterable over minibatches, each minibatch being a
			LongTensor of shape (batch_size, n_attributes).

		"""

		self.eval()
		device = self.c.device
		embeddings = []
		with torch.no_grad():
			for inputs in dataloader:
				inputs = inputs.to(device)
				embeddings.append(
					self._hidden_embedding(inputs).mean(0).cpu()
				)
		self.c.data = (sum(embeddings)/len(embeddings))
		self.train()


	def _update_reg_coeff(self, loss, reg_loss):
		"""
		Computes the new weight of the additional regularization
		term, then stores the result in self.reg_coeff.

		Arguments
		---------
		loss : float or Tensor of shape (1,)
			SVDD loss for the current minibatch.
		reg_loss : float or Tensor of shape (1,)
			Regularization term for the current minibatch.

		"""

		self.reg_coeff *= self.alpha
		ratio = loss / (reg_loss + 1e-5)
		self.reg_coeff += self.beta * (1 - self.alpha) * ratio


class NoiseContrastiveEstimation(DeepCatAD):
	"""
	Implementation of NCE-based anomaly detection, following [1].
	The log-probability of a given sample is obtained by passing its
	hidden representation through a linear layer.
	The bias term of this linear layer acts as an estimate of the
	partition function (see [2] for a more detailed description of
	NCE).


	Parameters
	----------
	arities : iterable of shape (n_attributes,), default=None
		Interable of integers representing the number of possible
		values for each categorical attribute.
	architecture : str, default='dnn'
		Name of the neural architecture to use.
		Possible values: 'autoint', 'dnn', 'nfm'.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	hidden_dim : int, default=16
		Dimension :math:`d'` of the hidden representation space.
	hidden_layers : int, default=2
		Number of hidden layers of the neural architecture.
	n_heads : int, default=8
		Number of attention heads in the Multihead Attention blocks.
		Used for AutoInt only.
	dropout : float, default=0
		Dropout rate applied at each fully connected layer during
		training.
		Used for DNN and NFM only.
	residual_connections : bool, default=True
		Whether to add skip connections in fully connected networks.
		Used for DNN and NFM only.
	bias : bool, default=True
		Whether to include bias terms in linear layers.
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
	bce : nn.Module
		PyTorch module computing the binary cross-entropy loss.
	fc : nn.Module
		PyTorch module implementing the final linear layer, which
		computes the log-probability of an instance given its hidden
		representation.

	References
	----------
	[1] Chen, Ting, et al. "Entity embedding-based anomaly detection
		for heterogeneous categorical events". IJCAI, 2016.
	[2] Gutmann, Michael and Hyvärinen, Aapo. "Noise-contrastive
		estimation: A new estimation principle for unnormalized
		statistical models". AISTATS, 2010.

	"""

	def __init__(
		self,
		arities=None,
		architecture='dnn',
		embedding_dim=128,
		hidden_dim=16,
		hidden_layers=2,
		n_heads=8,
		dropout=0,
		residual_connections=True,
		bias=True,
		n_noise_samples=3,
		noise_dist=None,
		**kwargs
		):

		super(NoiseContrastiveEstimation, self).__init__(
			arities,
			embedding_dim=embedding_dim,
			hidden_dim=hidden_dim,
			architecture=architecture,
			hidden_layers=hidden_layers,
			n_heads=n_heads,
			dropout=dropout,
			residual_connections=residual_connections,
			bias=bias
		)

		self.n_noise_samples = n_noise_samples
		if noise_dist is not None:
			self.noise_dist = noise_dist
		else:
			self.noise_dist = [torch.ones(n)/n for n in arities]
		self.noise_dist = [
			nn.Parameter(nd) for nd in self.noise_dist
		]
		for nd in self.noise_dist:
			nd.requires_grad = False

		self.bce = nn.BCEWithLogitsLoss()
		self.fc = nn.Linear(hidden_dim, 1)


	def forward(self, inputs, **kwargs):
		"""
		Forward pass of the model.

		Arguments
		---------
		inputs : LongTensor of shape (batch_size, n_attributes)
			Input samples.

		Other arguments
		---------------
		**kwargs : not used.

		Returns
		-------
		y : Tensor
			If the model is in training mode, y has shape (1,) and
			contains the total loss for the input samples.
			Otherwise, y has shape (batch_size,) and contains the
			anomaly scores of the input samples.

		"""

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

		H = self._hidden_embedding(inputs)
		return self.fc(H).squeeze(1)


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


def make_model(model_type, **kwargs):
	"""
	Instantiates and returns an anomaly detector of the desired type.

	Arguments
	---------
	model_type : str
		Name of the desired model.
		Possible values: 'nce', 'svdd'.

	Other arguments
	---------------
	**kwargs : keyword arguments that are passed to the constructor of
		the model.

	Returns
	-------
	model : nn.Module
		Anomaly detector of the desired type and with the parameters
		given as input.

	"""

	if model_type == 'svdd':
		model = DeepCatSVDD(**kwargs)
	elif model_type == 'nce':
		model = NoiseContrastiveEstimation(**kwargs)
	else:
		raise ValueError('Unknown model type: {0}'.format(model_type))
	return model
