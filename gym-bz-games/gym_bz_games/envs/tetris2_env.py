import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_bz_games.envs.tetris2_ctl import controller

class TetrisEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=2, shape=(29, 10), dtype=np.float32)
        self.controller = controller()

    """
    action space
        0: move right
        1: move left
        2: rotate clockwise
        3: rotate counterclockwise
        4: soft drop
        5: hard drop
        6: hold
    """
    def step(self, action):
        reward = 0
        landed = False
        fire = 0
        pos = [0, 0]
        if action == 0:
            self.controller.move_x(1)
        elif action == 1:
            self.controller.move_x(-1)
        elif action == 2:
            self.controller.rotate(1)
        elif action == 3:
            self.controller.rotate(-1)
        elif action == 4 or action == 5:
            if action == 4:
                landed, reward, column_list, perfect_landed, pos = self.controller.soft_drop()
            elif action == 5:
                landed, reward, column_list, perfect_landed, pos = self.controller.hard_drop()
            fire = reward
            if reward > 0:
                reward += 2
            if landed:
                if perfect_landed:
                    reward += 1
                height = self.controller.highest()
                reward -= height * 0.01
        #elif action == 6:
        #    self.controller.hold()

        obs = np.array(self.controller.get_state(), dtype=np.float32)

        is_done = self.controller.gameover
        score = self.controller.score
        lines = self.controller.lines_deleted_all
        used = self.controller.total_used

        return obs, reward, is_done, {"landed":landed, "fire":fire, "pos":pos, "score":score, "lines": lines, "used": used}


    def reset(self):
        self.controller.reset()
        obs = np.array(self.controller.get_state(), dtype=np.float32)
        return obs

    def render(self, mode='human'):
        pass
 
    def close(self):
        pass

    def add_fire(self, fire):
        self.controller.add_fire(fire)
