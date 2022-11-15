import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler, autocast

from dcad.models import make_model
from utils.data import duplicate_samples
from utils.estimators import CategoricalAD




class DCAD(CategoricalAD):
	"""
	Scikit-learn wrapper for the Deep Categorical Anomaly Detection
	framework.
	See dcad.models.DeepCatAD for the actual implementation of the
	framework.


	Parameters
	----------
	arities : interable of shape (n_attributes,)
		Interable of integers representing the number of possible
		values for each categorical attribute.
	model_type : str, default='nce'
		Anomaly detection algorithm to use.
		Possible values: 'nce', 'svdd'.
	architecture : str, default='dnn'
		Neural architecture to use.
		Possible values: 'dnn', 'nfm', 'autoint'.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	hidden_dim : int, default=16
		Dimension :math:`d'` of the hidden representation space.
	hidden_layers : int, default=2
		Number of hidden layers of the neural architecture.
	reg : str, default=None
		Type of regularization to use for DeepSVDD (in addition to
		weight decay and dropout).
		Possible values: 'noise', 'variance'.
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
		Number of negative samples drawn per attribute for each
		positive sample.
		Used for NCE only.
	nu : float, default=.01
		Soft boundary hyperparameter.
		Used for DeepSVDD only.
	warmup_epochs : int, default=10
		Number of initial training epochs to perform before updating
		the radius of the data-enclosing hypersphere.
		Used for DeepSVDD only.
	alpha : float, default=.9
		Controls how quickly the relative weight of the additional
		regularization term in the overall loss changes.
		With alpha=1, the weight is constant.
		With alpha=0, the weight is entirely recomputed at each
		minibatch, based on the ratio between the SVDD loss and the
		regularization term for the current minibatch.
		Used only for DeepSVDD, when reg is not None.
	beta : float, default=.5
		Controls the ratio between the SVDD loss and the additional
		regularization term.
		Used only for DeepSVDD, when reg is not None.
	n_noise_tasks : int, default=10
		Number of noisy classification tasks used to compute the
		noise regularization term.
		Used only for DeepSVDD, when reg=='noise'.
	variance_threshold : float, default=.1
		When using variance-based regularization, the additional
		regularization term is added only when the total sample
		variance of the hidden features for the current minibatch
		falls below this threshold.
		Used only for DeepSVDD, when reg=='variance'.
	annealing_rate : int, default=3
		Number of epochs between two decreases of the variance
		threshold.
		Used only for DeepSVDD, when reg=='variance'.
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
		PyTorch module implementing the neural architecture and the
		anomaly scoring function.
	
	"""

	def __init__(
		self,
		arities,
		model_type='nce',
		architecture='dnn',
		embedding_dim=128,
		hidden_dim=16,
		hidden_layers=2,
		reg=None,
		n_heads=8,
		dropout=0,
		residual_connections=True,
		bias=True,
		n_noise_samples=3,
		nu=.01,
		warmup_epochs=10,
		alpha=.9,
		beta=.5,
		n_noise_tasks=10,
		variance_threshold=.1,
		annealing_rate=3,
		batch_size=128,
		epochs=20,
		lr=1e-3,
		weight_decay=1e-5,
		device=torch.device('cuda')
		):

		self.arities = arities
		self.model_type = model_type
		self.architecture = architecture
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.hidden_layers = hidden_layers
		self.reg = reg
		self.n_heads = n_heads
		self.dropout = dropout
		self.residual_connections = residual_connections
		self.bias = bias
		self.nu = nu
		self.warmup_epochs = warmup_epochs
		self.n_noise_samples = n_noise_samples
		self.alpha = alpha
		self.beta = beta
		self.n_noise_tasks = n_noise_tasks
		self.variance_threshold = variance_threshold
		self.annealing_rate = annealing_rate

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
			self.arities[j] for j=0,...,n_attributes-1.
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

		self.model = make_model(
			self.model_type,
			arities=self.arities,
			embedding_dim=self.embedding_dim,
			hidden_dim=self.hidden_dim,
			hidden_layers=self.hidden_layers,
			architecture=self.architecture,
			reg=self.reg,
			n_heads=self.n_heads,
			dropout=self.dropout,
			residual_connections=self.residual_connections,
			bias=self.bias,
			nu=self.nu,
			n_noise_samples=self.n_noise_samples,
			noise_dist=ndist,
			alpha=self.alpha,
			beta=self.beta,
			num_noise_tasks=self.n_noise_tasks,
			variance_threshold=self.variance_threshold,
			annealing_rate=self.annealing_rate
		).to(self.device)

		self.losses = []

		if self.model == 'svdd':
			dataloader = DataLoader(
				dataset,
				batch_size=10000,
				shuffle=False,
				pin_memory=True
			)
			self.model.set_center(dataloader)

		self.model.train()

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
		for epoch in range(self.epochs):
			running_loss = 0
			for inputs in dataloader:
				inputs = inputs.to(self.device)
				with autocast():
					loss = self.model(
						inputs,
						update_R=(epoch >= self.warmup_epochs)
					)
				running_loss += loss.data.item()
				optimizer.zero_grad()
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			self.losses.append(running_loss)
			if self.reg == 'variance' and self.model_type == 'svdd':
				self.model.regularizer.update()

		return self
