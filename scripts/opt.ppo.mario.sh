python baselines3-zoo/train.py --algo qrdqn --env BZ-Mario-v0  -n 200000 -optimize --n-trials 2000 --sampler tpe --pruner median --vec-env=subproc --num-threads=4 --n-jobs 4
