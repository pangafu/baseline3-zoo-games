#python train.py --algo ppo --env CartPole-v1
#python train.py --algo ppo --env PongNoFrameskip-v4 
#python train.py --algo ppo --env BreakoutNoFrameskip-v4 --save-freq 10000
#python train.py --algo ppo --env SuperMarioBros-v0 --save-freq 10000 --n-jobs=8

#python train.py --algo ppo --env TetrisA-v0 --save-freq 20480 --eval-freq 20480 --eval-episodes=2 --vec-env=subproc --num-threads=8

python train.py --algo ppo --env EnvMarioLocal-v0 --save-freq 20480 --eval-freq 20480 --eval-episodes=2 --vec-env=subproc --num-threads=8
