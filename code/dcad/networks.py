import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.modules import MLP, PairwiseInteractions, SAB




class AutoInt(nn.Module):
	"""
	AutoInt architecture, after [1].
	This architecture uses Transformer-like self-attention layers [2]
	to compute hidden representations of the attributes.
	The final hidden representation of the instance is then obtained
	by passing the concatenation of the attribute embeddings through a
	linear layer.


	Parameters
	----------
	n_inputs : int, default=None
		Number of attributes per instance.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	hidden_dim : int, default=16
		Dimension :math:`d'` of the hidden representation space.
	num_layers : int, default=2
		Number of self-attention layers.
	n_heads : int, default=8
		Number of attention heads in each self-attention layer.
	bias : bool, default=True
		Whether to include a bias term in the final linear layer.

	Attributes
	----------
	layers : nn.ModuleList of shape (num_layers,)
		List of modules, each of which represents one self-attention
		layer.
	fc : nn.Module
		PyTorch module implementing the final linear layer.

	References
	----------
	[1] Song, Weiping, et al. "Autoint: Automatic feature interaction
		learning via self-attentive neural networks". CIKM, 2019.
	[2] Vaswani, Ashish, et al. "Attention is all you need".
		NeurIPS, 2017.

	"""

	def __init__(
		self,
		n_inputs=None,
		embedding_dim=128,
		hidden_dim=16,
		num_layers=2,
		n_heads=8,
		bias=True,
		**kwargs
		):

		super(AutoInt, self).__init__()

		self.layers = nn.ModuleList([
			SAB(embedding_dim, n_heads)
			for i in range(num_layers)
		])
		self.fc = nn.Linear(
			n_inputs * embedding_dim, hidden_dim, bias=bias
		)


	def forward(self, X):
		"""
		Forward pass of the model.

		Arguments
		---------
		X : Tensor of shape (batch_size, n_attributes, embedding_dim)
			Input samples.

		Returns
		-------
		H : Tensor of shape (batch_size, hidden_dim)
			Hidden representations of the input samples.

		"""

		b, m, d = X.shape
		H = X
		for l in self.layers:
			H = l(H)
		H = H.reshape(b, m * d)
		return self.fc(H)


class DNN(nn.Module):
	"""
	Fully connected neural architecture, similar to the deep component
	of Wide&Deep [1] and DeepFM [2].
	The embeddings of the attributes are simply concatenated and passed
	through a fully connected network.
	The hidden layers have the same number of units as the input layer,
	and the output layer returns the hidden representation.


	Parameters
	----------
	n_inputs : int, default=None
		Number of attributes per instance.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	hidden_dim : int, default=16
		Dimension :math:`d'` of the hidden representation space.
	num_layers : int, default=2
		Number of hidden layers.
	dropout : float, default=0
		Dropout rate to apply at each hidden layer.
	residual_connections : bool, default=True
		Whether to add skip connections at each hidden layer.
	bias : bool, default=True
		Whether to include bias terms in the linear layers.

	Attributes
	----------
	nn : nn.Module
		PyTorch module implementing the fully connected neural network.

	References
	----------
	[1] Cheng, Heng-Tze, et al. "Wide & deep learning for recommender
		systems". DLRS, 2016.
	[2] Guo, Huifeng, et al. "DeepFM: a factorization-machine based
		neural network for CTR prediction". IJCAI, 2017.

	"""

	def __init__(
		self,
		n_inputs=None,
		embedding_dim=128,
		hidden_dim=16,
		num_layers=2,
		dropout=0,
		residual_connections=True,
		bias=True,
		**kwargs
		):

		super(DNN, self).__init__()

		self.nn = MLP(
			n_inputs * embedding_dim,
			hidden_dim,
			num_layers=num_layers,
			dropout=dropout,
			residual_connections=residual_connections,
			bias=bias
		)


	def forward(self, X):
		"""
		Forward pass of the model.

		Arguments
		---------
		X : Tensor of shape (batch_size, n_attributes, embedding_dim)
			Input samples.

		Returns
		-------
		H : Tensor of shape (batch_size, hidden_dim)
			Hidden representations of the input samples.

		"""

		b, m, d = X.shape
		H = X.reshape(b, m * d)
		return self.nn(H)


class NFM(nn.Module):
	"""
	Neural Factorization Machine architecture, after [1].
	This architecture first computes all pairwise Hadamard products
	between the attribute embeddings, then passes their sum through
	a fully connected neural network.
	The output of this network is the final hidden representation.


	Parameters
	----------
	n_inputs : int, default=None
		Number of attributes per instance.
	embedding_dim : int, default=128
		Dimension :math:`d` of the attribute embedding space.
	hidden_dim : int, default=16
		Dimension :math:`d'` of the hidden representation space.
	num_layers : int, default=2
		Number of hidden layers.
	dropout : float, default=0
		Dropout rate to apply at each hidden layer.
	residual_connections : bool, default=True
		Whether to add skip connections at each hidden layer.
	bias : bool, default=True
		Whether to include bias terms in the linear layers.

	Attributes
	----------
	pi : nn.Module
		PyTorch module computing all pairwise products between the
		attribute embeddings.
	nn : nn.Module
		PyTorch module implementing the fully connected neural network.

	References
	----------
	[1] He, Xiangnan and Chua, Tat-Seng. "Neural factorization machines
		for sparse predictive analytics". SIGIR, 2017.

	"""

	def __init__(
		self,
		n_inputs=None,
		embedding_dim=128,
		hidden_dim=16,
		num_layers=2,
		dropout=0,
		residual_connections=True,
		bias=True,
		**kwargs
		):

		super(NFM, self).__init__()

		self.pi = PairwiseInteractions(n_inputs)
		self.nn = MLP(
			embedding_dim,
			hidden_dim,
			num_layers=num_layers,
			dropout=dropout,
			residual_connections=residual_connections,
			bias=bias
		)


	def forward(self, X):
		"""
		Forward pass of the model.

		Arguments
		---------
		X : Tensor of shape (batch_size, n_attributes, embedding_dim)
			Input samples.

		Returns
		-------
		H : Tensor of shape (batch_size, hidden_dim)
			Hidden representations of the input samples.

		"""

		H = self.pi(X)
		H = H.sum(1)
		return self.nn(H)


def make_network(architecture, **kwargs):
	"""
	Instantiates and returns a network with the desired architecture.

	Arguments
	---------
	architecture : str
		Name of the desired architecture.
		Possible values: 'autoint', 'dnn', 'nfm'.

	Other arguments
	---------------
	**kwargs : keyword arguments that are passed to the constructor of
		the network.

	Returns
	-------
	net : nn.Module
		Neural network with the desired architecture and parameters.

	"""

	if architecture == 'autoint':
		net = AutoInt(**kwargs)
	elif architecture == 'dnn':
		net = DNN(**kwargs)
	elif architecture == 'nfm':
		net = NFM(**kwargs)
	else:
		raise ValueError(
			'Unknown architecture type: {0}'.format(architecture)
		)
	return net
