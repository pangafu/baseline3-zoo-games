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
    ./scripts/train.ppo.mario.sh
    
    
# Test
    ./scripts/test.ppo.mario.sh
    
    
# Clear (clear all trained model, beware to use!)
    ./scripts/test.ppo.mario.sh
    
# Result
|  Game            | PPO  | QRDQN |
| ---------------  | ---- |  ---- | 
| BZ-Tetris2-v0  | [well](https://github.com/pangafu/baseline3-zoo-games-logs/tree/main/ppo/BZ-Tetris2-v0_1/) |  - |
| BZ-Mario-1-1-v0  | [pass](https://github.com/pangafu/baseline3-zoo-games-logs/tree/main/ppo/BZ-Mario-1-1-v0_1) |  - |
| BZ-Mario-1-2-v0  | [pass](https://github.com/pangafu/baseline3-zoo-games-logs/tree/main/ppo/BZ-Mario-1-2-v0_1) |  - |

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
