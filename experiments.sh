cd code/
for i in {0..9}; do
	python run_experiment.py --input_dir ../data/ids2017/ --output_dir ../results/ids2017/ape/ --model ape --embedding_dim 32 64 128 --batch_size 512 --epochs 10 --seed $i;
	python run_experiment.py --input_dir ../data/ids2017/ --output_dir ../results/ids2017/ptf/ --model ptf --embedding_dim 32 64 128 --seed $i;
	python run_experiment.py --input_dir ../data/ids2017/ --output_dir ../results/ids2017/iforest/ --model iforest --seed $i;
	for arch in dnn nfm; do
		python run_experiment.py --input_dir ../data/ids2017/ --output_dir ../results/ids2017/nce_$arch/ --model nce --architecture $arch --embedding_dim 32 64 128 --hidden_dim 16 32 --layers 2 3 4 --dropout 0 0.1 0.2 --batch_size 512 --epochs 10 --seed $i;
		python run_experiment.py --input_dir ../data/ids2017/ --output_dir ../results/ids2017/svdd_$arch/ --model svdd --architecture $arch --embedding_dim 32 64 128 --hidden_dim 16 32 --layers 2 3 4 --dropout 0 0.1 0.2 --regularization noise --num_noise_tasks 100 --batch_size 512 --epochs 10 --seed $i;
	done
	python run_experiment.py --input_dir ../data/ids2017/ --output_dir ../results/ids2017/nce_autoint/ --model nce --architecture autoint --embedding_dim 32 64 128 --hidden_dim 16 32 --layers 2 3 4 --batch_size 512 --epochs 10 --seed $i;
	python run_experiment.py --input_dir ../data/ids2017/ --output_dir ../results/ids2017/svdd_autoint/ --model svdd --architecture autoint --embedding_dim 32 64 128 --hidden_dim 16 32 --layers 2 3 4 --regularization noise --num_noise_tasks 100 --batch_size 512 --epochs 10 --seed $i;

	python run_experiment.py --input_dir ../data/unsw_nb15/ --output_dir ../results/unsw_nb15/ape/ --model ape --embedding_dim 32 64 128 --batch_size 512 --epochs 20 --seed $i;
	python run_experiment.py --input_dir ../data/unsw_nb15/ --output_dir ../results/unsw_nb15/ptf/ --model ptf --embedding_dim 32 64 128 --seed $i;
	python run_experiment.py --input_dir ../data/unsw_nb15/ --output_dir ../results/unsw_nb15/iforest/ --model iforest --seed $i;
	for arch in dnn nfm; do
		python run_experiment.py --input_dir ../data/unsw_nb15/ --output_dir ../results/unsw_nb15/nce_$arch/ --model nce --architecture $arch --embedding_dim 32 64 128 --hidden_dim 16 32 --layers 2 3 4 --dropout 0 0.1 0.2 --batch_size 512 --epochs 20 --seed $i;
		python run_experiment.py --input_dir ../data/unsw_nb15/ --output_dir ../results/unsw_nb15/svdd_$arch/ --model svdd --architecture $arch --embedding_dim 32 64 128 --hidden_dim 16 32 --layers 2 3 4 --dropout 0 0.1 0.2 --regularization noise --num_noise_tasks 100 --batch_size 512 --epochs 20 --seed $i;
	done
	python run_experiment.py --input_dir ../data/unsw_nb15/ --output_dir ../results/unsw_nb15/nce_autoint/ --model nce --architecture autoint --embedding_dim 32 64 128 --hidden_dim 16 32 --layers 2 3 4 --batch_size 512 --epochs 20 --seed $i;
	python run_experiment.py --input_dir ../data/unsw_nb15/ --output_dir ../results/unsw_nb15/svdd_autoint/ --model svdd --architecture autoint --embedding_dim 32 64 128 --hidden_dim 16 32 --layers 2 3 4 --regularization noise --num_noise_tasks 100 --batch_size 512 --epochs 20 --seed $i;
done