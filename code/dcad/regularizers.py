import torch
import torch.nn as nn
from torch.nn import functional as F




class NoiseRegularizer(nn.Module):
	"""
	Implementation of noise-based regularization for DeepSVDD, based
	on [1].
	This regularizer generates num_tasks random binary labels for each
	sample.
	It then tries to predict these labels based on the hidden
	representation of the sample, and the corresponding cross-entropy
	loss is used as a regularization term.
	The rationale is that trying to predict random noise prevents the
	neural network from mapping all instances to the same hidden
	representation.


	Parameters
	----------
	in_channels : int
		Dimension :math:`d'` of the hidden representation space.
	num_tasks : int, default=10
		Number of random labels to predict for each sample.

	Attributes
	----------
	fc : nn.Module
		PyTorch module implementing the linear prediction layer.

	References
	----------
	[1] Chong, Penny, et al. "Simple and effective prevention of mode
		collapse in deep one-class classification". IJCNN, 2020.

	"""

	def __init__(
		self,
		in_channels,
		num_tasks=10
		):

		super(NoiseRegularizer, self).__init__()
		self.fc = nn.Linear(in_channels, num_tasks)


	def forward(self, X):
		"""
		Forward pass of the model.

		Arguments
		---------
		X : Tensor of shape (batch_size, hidden_dim)
			Hidden representations of the input samples.

		Returns
		-------
		reg : Tensor of shape (1,)
			Regularization term for the minibatch.

		"""

		H = self.fc(X)
		y = torch.randint_like(H, 0, 2)
		return F.binary_cross_entropy_with_logits(H, y.float())


class VarianceRegularizer(nn.Module):
	"""
	Implementation of variance-based regularization for DeepSVDD, based
	on [1].
	This regularizer prevents the total sample variance of the hidden
	features from falling below a threshold.
	This prevents the hidden representations to collapse onto a single
	point.


	Parameters
	----------
	initial_threshold : float, default=.1
		Initial variance threshold below which the regularization term
		becomes nonzero.
	annealing_rate : int, default=3
		Number of epochs between two decreases of the variance
		threshold.

	Attributes
	----------
	ep : int
		Current training epoch.

	References
	----------
	[1] Chong, Penny, et al. "Simple and effective prevention of mode
		collapse in deep one-class classification". IJCNN, 2020.

	"""

	def __init__(
		self,
		initial_threshold=.1,
		annealing_rate=3
		):

		super(VarianceRegularizer, self).__init__()
		self.d = initial_threshold
		self.r = annealing_rate
		self.ep = 0


	def forward(self, X):
		"""
		Forward pass of the model.

		Arguments
		---------
		X : Tensor of shape (batch_size, hidden_dim)
			Hidden representations of the input samples.

		Returns
		-------
		reg : Tensor of shape (1,)
			Regularization term for the minibatch.

		"""

		V = X.var()
		return torch.fmax(
				self.d - V,
				torch.zeros_like(V)
			)


	def update(self):
		"""
		Increments the index of the current epoch.
		If the number of epochs since the last update of the variance
		threshold is equal to annealing_rate, the variance threshold
		is divided by 10.

		"""

		self.ep += 1
		if self.ep % self.r == 0:
			self.d /= 10
