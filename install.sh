# git clone submodule baselines3-zoo
git submodule update --init

# install baselines3-zoo
cd baselines3-zoo
pip install -r requirements.txt

# install gym-bz-games
cd ../gym-bz-games
pip install -e .