import os
import gym
import numpy as np
import cv2
from gym import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym.spaces import Box
from gym.spaces.discrete import Discrete
import random
from gym_bz_games.envs.tetris2_env import TetrisEnv
import time

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
        self.env = TetrisEnv()

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.last_state = None
        self.last_reward = 0
        self.last_done = False
        self.last_info = None

        self.max_reward = -100
        self.curr_reward_sum = 0

        #print(self.action_space)
        #print(self.observation_space)


    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.last_state = state
        self.last_reward = reward
        self.last_done = done
        self.last_info = info


        # reward sum
        self.curr_reward_sum += reward

        if self.curr_reward_sum > self.max_reward:
            self.max_reward = self.curr_reward_sum
            print(">> MAX REWARD : reward:{}".format(self.max_reward))

        return state[:, :, None], reward, done, info

    def reset(self):
        self.last_state = self.env.reset()
        self.last_reward = 0
        self.last_done = False
        self.last_info = None

        self.curr_reward_sum = 0

        return self.last_state[:, :, None]

    def render(self, mode='human'):
        if self.last_reward != 0:
            print("-------------------------------------------------------------")
            print(self.last_state)
            print("Reward:{}".format(self.last_reward))
            print("Done:{}".format(self.last_done))
            print("Info:{}".format(self.last_info))
            time.sleep(0.3)
        else:
            return

    def close (self):
        return self.env.close()

    def print_status(self):
        return print(self.env)
