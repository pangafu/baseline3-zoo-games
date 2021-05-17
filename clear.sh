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

rm -rf logs/$1/$2*
