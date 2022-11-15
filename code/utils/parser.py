import argparse




def make_argument_parser():
	parser = argparse.ArgumentParser()

	# I/O arguments
	parser.add_argument(
		'--input_dir',
		required=True,
		help='Path to the input files.'
	)
	parser.add_argument(
		'--output_dir',
		default=None,
		help='Path where the output should be written (if None, input'
			 ' directory is used).'
	)

	# Specification of the model and neural architecture (if applicable)
	parser.add_argument(
		'--model',
		default='nce',
		help='Anomaly detection algorithm to use.'
			 'Possible values: nce, svdd, ape, ptf, iforest.'
	)
	parser.add_argument(
		'--architecture',
		default='dnn',
		help='Neural architecture to use. '
			 'Used only when model is nce or svdd. '
			 'Possible values: dnn, nfm, autoint.'
	)

	# Tunable hyperparameters
	parser.add_argument(
		'--embedding_dim',
		type=int,
		nargs='+',
		default=[128],
		help='List of possible values for the dimension of the '
			 'attribute embedding space. '
			 'The best value is selected through cross-validation.'
	)
	parser.add_argument(
		'--hidden_dim',
		type=int,
		nargs='+',
		default=[16],
		help='List of possible values for the dimension of the '
			 'hidden representation space.'
			 'The best value is selected through cross-validation.'
			 'Used only when model is nce or svdd.'
	)
	parser.add_argument(
		'--layers',
		type=int,
		nargs='+',
		default=[2],
		help='List of possible values for the number of hidden layers'
			 ' of the neural architecture. '
			 'The best value is selected through cross-validation. '
			 'Used only when model is nce or svdd.'
	)
	parser.add_argument(
		'--neg_samples',
		type=int,
		nargs='+',
		default=[3],
		help='List of possible values for the number of negative '
			 'samples drawn per attribute. '
			 'The best value is selected through cross-validation. '
			 'Used only when model is nce, or ape.'
	)
	parser.add_argument(
		'--dropout',
		type=float,
		nargs='+',
		default=[0],
		help='List of possible values for the dropout rate. '
			 'The best value is selected through cross-validation. '
			 'Used only when model is nce or svdd and architecture is'
			 ' dnn or nfm.'
	)
	parser.add_argument(
		'--n_heads',
		type=int,
		nargs='+',
		default=[8],
		help='List of possible values for the number of attention '
			 'heads in the multihead attention layers. '
			 'The best value is selected through cross-validation. '
			 'Used only when model is nce or svdd and architecture is'
			 ' autoint.'
	)

	# Training-related hyperparameters
	parser.add_argument(
		'--epochs',
		type=int,
		default=20,
		help='Number of training epochs.'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=128,
		help='Size of the training minibatches.'
	)
	parser.add_argument(
		'--lr',
		type=float,
		default=1e-3,
		help='Learning rate for stochastic gradient descent.'
	)
	parser.add_argument(
		'--weight_decay',
		type=float,
		default=1e-5,
		help='Weight decay coefficient for the Adam optimizer.'
	)

	# SVDD-specific hyperparameters
	parser.add_argument(
		'--regularization',
		default=None,
		help='Type of additional regularization to use for DeepSVDD. '
			 'Used only when model is svdd. '
			 'Possible values: noise, variance.'
	)
	parser.add_argument(
		'--warmup_epochs',
		type=int,
		default=10,
		help='Number of initial training epochs to perform before '
			 'updating the radius of the data-enclosing hypersphere. '
			 'Used only when model is svdd.'
	)
	parser.add_argument(
		'--nu',
		type=float,
		default=.01,
		help='Soft boundary hyperparameter for DeepSVDD. '
			 'Used only when model is svdd.'
	)
	parser.add_argument(
		'--alpha',
		type=float,
		default=.9,
		help='Controls how quickly the relative weight of the '
			 'additional regularization term in the overall loss '
			 'changes. '
			 'Used only when model is svdd.'
	)
	parser.add_argument(
		'--beta',
		type=float,
		default=.5,
		help='Controls the ratio between the SVDD loss and the '
			 'additional regularization term. '
			 'Used only when model is svdd.'
	)
	parser.add_argument(
		'--n_noise_tasks',
		type=int,
		default=100,
		help='Number of noisy classification tasks used to compute '
			 'the noise regularization term. '
			 'Used only when model is svdd and regularization is '
			 'noise.'
	)
	parser.add_argument(
		'--variance_threshold',
		type=float,
		default=.1,
		help='Initial variance threshold below which the '
			 'regularization term becomes nonzero. '
			 'Used only when model is svdd and regularization is '
			 'variance.'
	)
	parser.add_argument(
		'--annealing_rate',
		type=int,
		default=3,
		help='Number of training epochs between two decreases of the '
			 'variance threshold. '
			 'Used only when model is svdd and regularization is '
			 'variance.'
	)

	# Others
	parser.add_argument(
		'--seed',
		type=int,
		default=0,
		help='Fixed seed for the RNG (for reproducibility).'
	)

	return parser