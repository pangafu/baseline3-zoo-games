if [ ! $1 ]; then
  echo "algo is blank, Please Input the 1 parameter."
  exit 1
else
  echo "algo is $1"
fi 

if [ ! $2 ]; then
  echo "env is blank, Please Input the 2 parameter."
  exit 1
else
  echo "env is $2"
fi 


python baselines3-zoo/train.py --algo $1 --env $2 --save-freq 20480 --eval-freq 20480 --eval-episodes=2 --vec-env=subproc --num-threads=8
