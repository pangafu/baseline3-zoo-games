python baselines3-zoo/train.py --algo ppo --env BZ-Tetris-v0  -n 100000 -optimize --n-trials 2000 --sampler tpe --pruner median --vec-env=subproc --num-threads=4 --n-jobs 4
