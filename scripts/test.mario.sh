#python enjoy.py --algo ppo --env CartPole-v1 --folder logs/
#python enjoy.py --algo ppo --env PongNoFrameskip-v4 --folder logs/

while true; do
  #python enjoy.py --algo ppo --env TetrisA-v0 --folder logs/ --load-best

  echo "-------------------------------------"
  echo ">>>> TEST LAST BEGIN + "
  echo ""

  python enjoy.py --algo ppo --env EnvMarioLocal-v0 --folder logs/ --num-threads 1 --n-envs 1 -n 2000 --load-checkpoint-last

  echo ""
  echo ">>>> TEST LAST END - "


  sleep 5

  echo ""

  echo ">>>> TEST BEST BEGIN + "
  echo ""

  python enjoy.py --algo ppo --env EnvMarioLocal-v0 --folder logs/ --num-threads 1 --n-envs 1 -n 2000 --load-best

  echo ""
  echo ">>>> TEST LAST END - "


  echo ""

  echo ">>>> DELETE BEGIN + "
  echo ""

  find logs/ppo/EnvMarioLocal-v0*/ -mmin +30 -name "rl*.zip"

  find logs/ppo/EnvMarioLocal-v0*/ -mmin +30 -name "rl*.zip" -exec rm -rf {} \;


  echo ""
  echo ">>>> DELETE END - "


  echo "-------------------------------------"
  echo ""
  echo ">>>> SLEEPING 240 SECOND"

  sleep 240;

  echo ""
done
