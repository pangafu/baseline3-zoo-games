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

echo "-------------------------------------"
echo ">>>> RECORDING BEST BEGIN + "
echo ""

export BZ_RECORD = 1

python baselines3-zoo/enjoy.py --algo $1 --env $2 --folder logs/ --num-threads 1 --n-envs 1 -n 2000 --load-checkpoint-last

unset BZ_RECORD

echo ""
echo ">>>> RECORDING BEST END - "