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


while true; do

  echo "-------------------------------------"
  echo ">>>> TEST LAST BEGIN + "
  echo ""

  export BZ_TEST=1

  python baselines3-zoo/enjoy.py --algo $1 --env $2 --folder logs/ --num-threads 1 --n-envs 1 -n 2000 --load-checkpoint-last

  echo ""
  echo ">>>> TEST LAST END - "


  sleep 5

  echo ""

  echo ">>>> TEST BEST BEGIN + "
  echo ""

  python baselines3-zoo/enjoy.py --algo $1 --env $2 --folder logs/ --num-threads 1 --n-envs 1 -n 2000 --load-best

  unset BZ_TEST
  echo ""
  echo ">>>> TEST BEST END - "


  echo ""

  echo ">>>> DELETE BEGIN + "
  echo ""

  find logs/ppo/$2*/ -mmin +60 -name "rl*.zip"

  find logs/ppo/$2*/ -mmin +60 -name "rl*.zip" -exec rm -rf {} \;


  echo ""
  echo ">>>> DELETE END - "


  echo "-------------------------------------"
  echo ""
  echo ">>>> SLEEPING 240 SECOND"

  sleep 240;

  echo ""
done
