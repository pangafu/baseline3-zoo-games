# Concept
Use [Baseline3 Zoo framework](https://github.com/DLR-RM/rl-baselines3-zoo) train games like Super Mario Bros.,  Tetris etc....

# Example
All videos put in
https://github.com/pangafu/baseline3-zoo-games-videoes

## PPO - Super Mario Bros.
![ppo_SuperMarioBros-1-1](https://github.com/pangafu/baseline3-zoo-games-videoes/raw/main/ppo/SuperMarioBros-1-1-v0.gif)
![ppo_SuperMarioBros-1-2](https://github.com/pangafu/baseline3-zoo-games-videoes/raw/main/ppo/SuperMarioBros-1-2-v0.gif)

## PPO - Tetris2
![ppo - Tetris2](https://github.com/pangafu/baseline3-zoo-games-videoes/raw/main/ppo/Tetris2-v0.gif)

# games
I have make or wrap games bellow in gym-bz-games/
|  GameName            | INFO |
| ---------------  | ---- | 
| BZ-Mario-v0        |    default mario world 1 stage 1  | 
| BZ-Mario-X-Y-v0    |    mario world X: world 1-8    Y: stage:1-4  | 
| BZ-MarioRandom-v0  |    mario world random choose  | 
| BZ-Tetris-v0       |    nes tetris   | 
| BZ-Tetris2-v0      |    tetris game in python with drop info and full info(can clear line, score up/down ...)  | 
| BZ-Tetris3-v0      |    tetris game in python with drop info (without info maybe evolved some advanced strategies) | 
| BZ-Tetris4-v0      |    tetris game in python in direct drop mode(drop with pos, rotate action) | 


# models
All modeles put in
https://github.com/pangafu/baseline3-zoo-games-logs

# Install
    git clone https://github.com/pangafu/baseline3-zoo-games.git
    cd baseline3-zoo-games/
    ./install.sh

# Download Trained models and videoes

    git clone https://github.com/pangafu/baseline3-zoo-games-videoes.git
    git clone https://github.com/pangafu/baseline3-zoo-games-logs.git
    
# Train
    ./train.sh ppo BZ-Mario-v0 

# Train continue
    ./train.sh ppo BZ-Mario-v0 logs/ppo/BZ-Mario——v0_1/best_model.zip
    
# Test 
    ./test.sh ppo BZ-Mario-v0 
    
# Clear (clear all trained model, beware to use!)
    ./clear.sh ppo BZ-Mario-v0 
    
# Result
|  Game            | PPO  | QRDQN |
| ---------------  | ---- |  ---- | 
| BZ-Tetris2-v0  | [well](https://github.com/pangafu/baseline3-zoo-games-logs/tree/main/ppo/BZ-Tetris2-v0_1/) |  - |
| BZ-Mario-1-1-v0  | [pass](https://github.com/pangafu/baseline3-zoo-games-logs/tree/main/ppo/BZ-Mario-1-1-v0_1) |  - |
| BZ-Mario-1-2-v0  | [pass](https://github.com/pangafu/baseline3-zoo-games-logs/tree/main/ppo/BZ-Mario-1-2-v0_1) |  - |
| BZ-MarioRandom-v0  | - |  - |

# Citing the Project
To cite this repository in publications:

    @misc{baseline3-zoo-games,
      author = {YuanQuan},
      title = {RL Baselines3 Zoo Games},
      year = {2021},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/pangafu/baseline3-zoo-games}},
    }
