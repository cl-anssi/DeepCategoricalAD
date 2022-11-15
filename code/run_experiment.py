import argparse
import json
import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import IsolationForest

from utils.data import make_train_dataset, make_test_dataset
from utils.parser import make_argument_parser
from dcad.dcad import DCAD
from ape.ape import APE

from utils.estimators import TEST_BATCH_SIZE




TEST_BATCH_SIZE = 5000

parser = make_argument_parser()
args = parser.parse_args()


if args.output_dir is not None:
	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
else:
	output_dir = args.input_dir

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

print('[*] Reading dataset')
one_hot = (args.model == 'iforest')
fp = os.path.join(args.input_dir, 'train.csv')
dataset, y, encoders = make_train_dataset(
	fp,
	one_hot=one_hot
)
if one_hot:
	arities = [
		cat.shape[0] for cat in encoders.categories_
	]
else:
	arities = [len(e.classes_) for e in encoders]

print('[*] Building model')
if args.model == 'ape':
	hp_grid = {
		'embedding_dim': args.embedding_dim,
		'n_noise_samples': args.neg_samples
	}
	base_model = APE(
		arities,
		epochs=args.epochs,
		batch_size=args.batch_size,
		lr=args.lr,
		weight_decay=args.weight_decay,
		device=device
	)
elif args.model == 'ptf':
	# We only import PTF here as it depends on a specific
	# external library
	from ptf.ptf import PTF
	hp_grid = {
		'K': args.embedding_dim
	}
	base_model = PTF(random_state=args.seed)
	dataset = dataset[:, :-1]
	y = y * np.ones(dataset.shape[0])
elif args.model in ('nce', 'svdd'):
	hp_grid = {
		'embedding_dim': args.embedding_dim,
		'hidden_dim': args.hidden_dim,
		'hidden_layers': args.layers,
		'n_noise_samples': args.neg_samples,
		'dropout': args.dropout,
		'n_heads': args.n_heads
	}
	base_model = DCAD(
		arities,
		model_type=args.model,
		architecture=args.architecture,
		reg=args.regularization,
		epochs=args.epochs,
		batch_size=args.batch_size,
		device=device,
		nu=args.nu,
		warmup_epochs=args.warmup_epochs,
		alpha=args.alpha,
		beta=args.beta,
		n_noise_tasks=args.n_noise_tasks,
		variance_threshold=args.variance_threshold,
		annealing_rate=args.annealing_rate,
		lr=args.lr,
		weight_decay=args.weight_decay
	)
elif args.model == 'iforest':
	base_model = IsolationForest(random_state=args.seed)
else:
	raise ValueError('Unknown model type: %s' % args.model)

if args.model == 'iforest':
	model = base_model
else:
	model = GridSearchCV(
		base_model,
		hp_grid,
		verbose=4,
		error_score='raise'
	)

print('[*] Starting training')
model.fit(
	dataset,
	y if args.model == 'ptf' else None
)

try:
	res = {'loss': model.best_estimator_.losses}
except:
	res = {'loss': []}

print('[*] Reading test set')
fp = os.path.join(args.input_dir, 'test.csv')
dataset = make_test_dataset(
	fp,
	encoders,
	one_hot=one_hot
)

print('[*] Starting evaluation')
if one_hot:
	X, y = dataset
	preds = model.score_samples(X)
	res['scores'] = preds.tolist()
	res['labels'] = y.tolist()
else:
	dataloader = DataLoader(
		dataset,
		shuffle=False,
		batch_size=TEST_BATCH_SIZE,
		pin_memory=True
	)
	outputs = []
	targets = []
	with torch.no_grad():
		for inputs, labels in dataloader:
			outputs.append(-model.score_samples(inputs))
			targets.append(labels.squeeze(1).numpy())
	res['scores'] = np.concatenate(outputs).tolist()
	res['labels'] = np.concatenate(targets).tolist()

print('[*] Writing results')
if args.model in ('ape', 'ptf', 'iforest'):
	arch = 'na'
else:
	arch = args.architecture
fp = os.path.join(
	output_dir,
	'res_{0}_{1}_{2}.json'.format(
		args.model,
		arch,
		args.seed
	)
)
with open(fp, 'w') as out:
	out.write(json.dumps(res))
