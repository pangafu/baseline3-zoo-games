import os
import gym
import numpy as np
import cv2
from gym import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym.spaces import Box
from gym.spaces.discrete import Discrete
import random
from gym_bz_games.envs.tetris4_env import TetrisEnv
from gym_bz_games.wrappers import RecorderVideoTools
import time
from nes_py._image_viewer import ImageViewer
from gym.spaces import Discrete


# The last wrapped env
class Tetris4(gym.Env):
    metadata = {'render.modes':[ 'human', 'none','detail']}

    def __init__(self, height=20, width=10, block_size=10):
        self.height = height
        self.width = height
        self.block_size = block_size

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
        self.env = TetrisEnv(self.height, self.width, self.block_size)

        self.action_space = Discrete(self.width*4)
        self.observation_space =  Box(low=0, high=self.height, shape=(self.width*4, 5), dtype=np.uint8)

        self.viewer = None

        self.last_states = None
        self.last_states_full = None
        self.last_actions = None
        self.last_reward = 0
        self.last_done = False
        self.last_info = {}

        self.max_reward = -100
        self.curr_reward_sum = 0
        self.curr_clear_lines = 0

        if need_record:
            self.recorder = RecorderVideoTools(saved_path=os.path.join("videoes", bz_record_algo, "Tetris4-v0.gif"))
        self.has_recorded = False
        self.need_record = need_record
        self.min_record_length = 10
        self.record_done = False


    def get_last_states_actions(self):
        next_steps = self.env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        self.last_states = np.array(next_states)
        self.last_actions = np.array(next_actions)
        self.last_states_full = np.concatenate([self.last_states, np.zeros((self.width*4 - len(self.last_states),4))],axis=0)
        self.last_states_full = self.last_states_full.astype(np.uint8)
        self.last_states_full =  np.insert(self.last_states_full, 0, values=np.zeros(self.width*4), axis=1)

        for i in range(len(self.last_states)):
            self.last_states_full[i, 0] = i

        return self.last_states_full


    def get_action(self, index):
        action_index = index % len(self.last_actions)
        return self.last_actions[action_index]

    def step(self, action):

        reward, done = self.env.step(self.get_action(action))
        self.last_reward = reward
        self.last_done = done

        # reward sum
        self.curr_reward_sum += self.last_reward

        if self.curr_clear_lines < self.env.cleared_lines:
            print(">> CLEAR LINES : Score:{} Pieces:{} Lines:{}".format(self.env.score, self.env.tetrominoes, self.env.cleared_lines))
            self.curr_clear_lines = self.env.cleared_lines

        if self.curr_reward_sum > self.max_reward:
            self.max_reward = self.curr_reward_sum
            print(">> MAX REWARD : Reward:{} Score:{} Pieces:{} Lines:{}".format(self.max_reward, self.env.score, self.env.tetrominoes, self.env.cleared_lines))

        if self.need_record and not self.has_recorded:
            self.recorder.record(self.env.render())

        self.record_done = (self.env.cleared_lines > 100)

        return self.get_last_states_actions(), self.last_reward, self.last_done, self.last_info

    def reset(self):
        self.env.reset()
        self.last_reward = 0
        self.last_done = False
        self.last_info = {}

        self.curr_reward_sum = 0
        self.curr_clear_lines = 0

        if self.need_record and not self.has_recorded:
            if self.recorder.record_length > self.min_record_length and self.record_done:
                self.has_recorded = True
                self.recorder.save()
            else:
                if self.recorder.record_length > self.min_record_length:
                    print("Record frame is {} ( Min Length {}), Record Done is {}, continue recording!".format(self.recorder.record_length, self.min_record_length, self.record_done))
                self.recorder.reset()

        return self.get_last_states_actions()

    def render(self, mode='human'):
        # if self.last_reward != 0:
        if mode == 'human1':
            if self.viewer is None:
                self.viewer = ImageViewer( caption="Tetris4", height=self.height*self.block_size, width=self.width*self.block_size)

            self.viewer.show(self.env.render())
        elif mode == 'none':
            pass

        elif mode == 'RGBArray':
            return self.env.render()
        else:
            return

    def close (self):
        if self.viewer is not None:
            self.viewer.close()
        return self.env.close()

    def print_status(self):
        print(">> STATUS : Score:{} Pieces:{} Lines:{}".format(self.env.score, self.env.tetrominoes, self.env.cleared_lines))
