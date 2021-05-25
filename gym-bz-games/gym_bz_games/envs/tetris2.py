import os
import gym
import numpy as np
import cv2
from gym import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym.spaces import Box
from gym.spaces.discrete import Discrete
import random
from gym_bz_games.envs.tetris2_engine import TetrisEngine

class Tetris2(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        need_record = False
        need_test = False

        bz_record = os.environ.get('BZ_RECORD')
        bz_record_algo = os.environ.get('BZ_RECORD_ALGO')

        if bz_record and bz_record == "1":
            need_record = True

        bz_test = os.environ.get('BZ_TEST')

        if bz_test and bz_test == "1":
            need_test = True

        #init
        self.env = TetrisEngine(10,20)

        self.action_space = Discrete(self.env.nb_actions)
        self.observation_space = Box(low=0, high=1, shape=(20, 10), dtype=np.uint8)

        #print(self.action_space)
        #print(self.observation_space)


    def step(self, action):
        info, reward, done = self.env.step(action)
        state = np.rot90(self.env.board,-1)
        return state, reward, done, info

    def reset(self):
        self.env.clear()
        return np.rot90(self.env.board,-1)

    def render(self, mode='human'):
        return print(self.env)

    def close (self):
        return

    def print_status(self):
        return print(self.env)
