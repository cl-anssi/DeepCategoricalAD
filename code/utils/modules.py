import torch
import torch.nn as nn
from torch.nn import functional as F




class LinearWeightedAvg(nn.Module):
	"""
	Computes the weighted sum of a fixed number of input vectors,
	with learnable weights.


	Parameters
	----------
	n_inputs : int
		Number of input vectors.

	Attributes
	----------
	weights : Tensor of shape (n_inputs,)
		Learnable weights used to compute the weighted sum.

	"""

	def __init__(
		self,
		n_inputs
		):

		super(LinearWeightedAvg, self).__init__()
		self.weights = nn.Parameter(torch.randn(n_inputs))

	def forward(self, inputs):
		"""
		Forward pass.

		Arguments
		---------
		inputs : Tensor of shape (batch_size, n_inputs, dimension)
			Input samples.

		Returns
		-------
		outputs : Tensor of shape (batch_size, dimension)
			Weighted sums of the input samples.

		"""

		return torch.einsum('xyz,y->xz', inputs, self.weights)


class MLP(nn.Module):
	"""
	Fully connected neural network with ReLU activations.
	Each hidden has as many neurons as the input layer.
	Implements batch normalization, dropout and optional residual
	connections between hidden layers.


	Parameters
	----------
	in_channels : int
		Dimension of the input layer.
	out_channels : int
		Dimension of the output layer.
	num_layers : int
		Number of hidden layers.
	dropout : float, default=0
		Dropout rate applied at each hidden layer during training.
	residual_connections : bool, default=True
		Whether to add skip connections at each hidden layer.
	bias : bool, default=True
		Whether to include a bias term in linear layers.

	Attributes
	----------
	layers : iterable of shape (num_layers,)
		List of PyTorch modules implementing the hidden layers.
	bn : iterable of shape (num_layers,)
		List of PyTorch modules implementing batch normalization.
	fc : nn.Module
		PyTorch module implementing the final linear layer.

	"""

	def __init__(
		self,
		in_channels,
		out_channels,
		num_layers,
		dropout=0,
		residual_connections=True,
		bias=True
		):

		super(MLP, self).__init__()

		self.dropout = dropout
		self.residual_connections = residual_connections

		self.layers = nn.ModuleList([
			nn.Linear(in_channels, in_channels, bias=bias)
			for i in range(num_layers)
		])
		self.bn = nn.ModuleList([
			nn.BatchNorm1d(in_channels, affine=False)
			for i in range(num_layers)
		])
		self.fc = nn.Linear(
			in_channels, out_channels, bias=bias
		)


	def forward(self, X):
		"""
		Forward pass.

		Arguments
		---------
		X : Tensor of shape (batch_size, in_channels)
			Input samples.

		Returns
		-------
		H : Tensor of shape (batch_size, out_channels)
			Hidden representations of the input samples.

		"""

		H = X
		for i, l in enumerate(self.layers):
			Hl = F.dropout(
				H,
				self.dropout,
				training=self.training
			)
			Hl = l(Hl)
			Hl = self.bn[i](Hl)
			Hl = F.relu(Hl)
			if self.residual_connections:
				H = H + Hl
			else:
				H = Hl

		return self.fc(H)


class PairwiseInteractions(nn.Module):
	"""
	Computes all pairwise Hadamard products between the vectors given
	as input.


	Parameters
	----------
	n_inputs : int
		Number of input vectors.

	Attributes
	----------
	i0 : list
		Indices used to compute the pairwise products.
	i1 : list
		Indices used to compute the pairwise products.

	"""

	def __init__(
		self,
		n_inputs
		):

		super(PairwiseInteractions, self).__init__()
		ind = [(i, j) for i in range(n_inputs) for j in range(i)]
		self.i0 = [i for i, j in ind]
		self.i1 = [j for i, j in ind]


	def forward(self, X):
		"""
		Forward pass.

		Arguments
		---------
		X : Tensor of shape (batch_size, n_inputs, dimension)
			Input samples.

		Returns
		-------
		P : Tensor of shape
			(batch_size, n_inputs * (n_inputs-1)/2, dimension)
			Pairwise products between the input samples.

		"""

		X0 = X[:, self.i0, :]
		X1 = X[:, self.i1, :]
		return X0*X1


class SAB(nn.Module):
	"""
	Self-attention block, after [1].


	Parameters
	----------
	in_channels : int
		Dimension of the input samples.

	n_heads : int, default=8
		Number of attention heads.

	Attributes
	----------
	mh_attn : nn.Module
		Multihead attention block.

	fc1 : nn.Module
		Hidden linear layer.

	fc2 : nn.Module
		Output linear layer.

	ln1 : nn.Module
		Layer normalization for the hidden linear layer.

	ln2 : nn.Module
		Layer normalization for the output linear layer.

	References
	----------
	[1] Song, Weiping, et al. "Autoint: Automatic feature interaction
		learning via self-attentive neural networks". CIKM, 2019.

	"""

	def __init__(
		self,
		in_channels,
		n_heads=8
		):

		super(SAB, self).__init__()

		self.mh_attn = nn.MultiheadAttention(
			in_channels, n_heads, batch_first=True
		)
		self.fc1 = nn.Linear(in_channels, in_channels)
		self.fc2 = nn.Linear(in_channels, in_channels)

		self.ln1 = nn.LayerNorm(in_channels)
		self.ln2 = nn.LayerNorm(in_channels)

	def forward(self, inputs):
		"""
		Forward pass.

		Arguments
		---------
		inputs : Tensor of shape (batch_size, n_inputs, dimension)
			Input samples.

		Returns
		-------
		H : Tensor of shape (batch_size, n_inputs, dimension)
			New representations of the input samples.

		"""

		X, _ = self.mh_attn(inputs, inputs, inputs, need_weights=False)
		H = self.ln1(inputs + X)
		return self.ln2(H + self.fc2(F.relu(self.fc1(H))))
