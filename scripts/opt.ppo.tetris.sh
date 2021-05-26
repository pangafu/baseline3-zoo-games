python baselines3-zoo/train.py --algo ppo --env BZ-Tetris2-v0  -n 200000 -optimize --n-trials 2000 --sampler tpe --pruner median --num-threads=4 --n-jobs 4
