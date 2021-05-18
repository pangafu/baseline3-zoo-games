# Concept
Use [Baseline3 Zoo framework](https://github.com/DLR-RM/rl-baselines3-zoo) train games like Super Mario Bros.,  Tetris etc....

# Example
All videos put in
https://github.com/pangafu/baseline3-zoo-games-videoes

## PPO - Super Mario Bros.
![ppo_SuperMarioBros-1-1](https://github.com/pangafu/baseline3-zoo-games-videoes/raw/main/ppo/SuperMarioBros-1-1-v0.gif)


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
    ./script/train.mario.sh
    
    
# Test
    ./script/test.mario.sh
    
    
# Clear (clear all trained model, beware to use!)
    ./script/test.mario.sh
    
# Result
|  Game            | PPO  | QRDQN |
| ---------------  | ---- |  ---- | 
| BZ-Mario-1-1-v0  | ok |  - |
| BZ-Mario-1-2-v0  | - |  - |
| BZ-Mario-1-3-v0  | - |  - |
| BZ-Mario-1-4-v0  | - |  - |
| BZ-Mario-2-1-v0  | - |  - |
| BZ-Mario-2-2-v0  | - |  - |
| BZ-Mario-2-3-v0  | - |  - |
| BZ-Mario-2-4-v0  | - |  - |
| BZ-Mario-3-1-v0  | - |  - |
| BZ-Mario-3-2-v0  | - |  - |
| BZ-Mario-3-3-v0  | - |  - |
| BZ-Mario-3-4-v0  | - |  - |
| BZ-Mario-4-1-v0  | - |  - |
| BZ-Mario-4-2-v0  | - |  - |
| BZ-Mario-4-3-v0  | - |  - |
| BZ-Mario-4-4-v0  | - |  - |
| BZ-Mario-5-1-v0  | - |  - |
| BZ-Mario-5-2-v0  | - |  - |
| BZ-Mario-5-3-v0  | - |  - |
| BZ-Mario-5-4-v0  | - |  - |
| BZ-Mario-6-1-v0  | - |  - |
| BZ-Mario-6-2-v0  | - |  - |
| BZ-Mario-6-3-v0  | - |  - |
| BZ-Mario-6-4-v0  | - |  - |
| BZ-Mario-7-1-v0  | - |  - |
| BZ-Mario-7-2-v0  | - |  - |
| BZ-Mario-7-3-v0  | - |  - |
| BZ-Mario-7-4-v0  | - |  - |
| BZ-Mario-8-1-v0  | - |  - |
| BZ-Mario-8-2-v0  | - |  - |
| BZ-Mario-8-3-v0  | - |  - |
| BZ-Mario-8-4-v0  | - |  - |
| BZ-Tetris-v0  | - |  - |

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
