import os
import gym
import numpy as np
import cv2
from gym import spaces
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym_bz_games.wrappers import CustomSkipFrame, NesFrameGray, NesFrameGrayHalf, RandomStart, RecorderVideo
import gym_super_mario_bros


class CustomReward(gym.Wrapper):
    def __init__(self, env: gym.Env, world, stage):
        gym.Wrapper.__init__(self, env)
        self.world = world
        self.stage = stage
        self.curr_score = 0
        self.curr_coins = 0
        self.curr_status = 0
        self.curr_x = 40
        self.curr_x_max = 40
        self.curr_x_max_frame = 0
        self.curr_frame = 0
        self.curr_reward_sum = 0
        self.max_reward = 0

    def step(self, action: int) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)

        reward = 0
        reward += (info["score"] - self.curr_score) / 20.
        self.curr_score = info["score"]

        reward += (info["coins"] - self.curr_coins) * 30.
        self.curr_coins = info["coins"]

        player_status = 0
        if info["status"] != "small" and info["status"] != "tall":
            print(info["status"])


        if info["status"] == "tall":
            player_status = 1

        if info["status"] != "small" and info["status"] != "tall":
            player_status = 2

        reward += (player_status - self.curr_status) * 200.
        self.curr_status = player_status

        self.curr_frame += 1

        self.curr_x = info["x_pos"]

        if self.curr_x > self.curr_x_max + max(min(self.curr_x_max/20, 100), 20):
            self.curr_x_max = self.curr_x_max + max(min(self.curr_x_max/20, 100), 20)
            reward += 2
            self.curr_x_max_frame = self.curr_frame


        if self.curr_frame - self.curr_x_max_frame > 3000:
            done = True



        if done:
            if info["flag_get"]:
                reward += 200
            else:
                reward -= 30

        self.curr_reward_sum += reward

        if self.curr_reward_sum > self.max_reward:
            self.max_reward = self.curr_reward_sum
            print(">> MAX REWARD : reward:{}  score:{}  coin:{}  status:{}  x:{}".format(self.max_reward, self.curr_score, self.curr_coins, self.curr_status, self.curr_x))


        return state, reward, done, info


    def reset(self, **kwargs) -> GymObs:
        self.curr_score = 0
        self.curr_coins = 0
        self.curr_status = 0
        self.curr_x = 40
        self.curr_x_max = 40
        self.curr_x_max_frame = 0
        self.curr_frame = 0
        self.curr_reward_sum = 0

        return self.env.reset(**kwargs)




class Mario(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, world = 1, stage = 1):
        self.world = world
        self.stage = stage
        
        need_record = False
        bz_record = os.environ.get('BZ_RECORD')
        if bz_record and bz_record == 1:
            need_record = True
        
        env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(self.world, self.stage))
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        
        if need_record:
            env = RecorderVideo(env, saved_path="videoes/")
        
        env = CustomSkipFrame(env, skip = 2)
        env = NesFrameGrayHalf(env)
        
        if not need_record:
            env = RandomStart(env, rnum = 5)
        
        env = CustomReward(env, self.world, self.stage)
        


        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        #print(self.action_space)
        #print(self.observation_space)


    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

    def close (self):
        return self.env.close()


class Mario11(Mario):
    def __init__(self):
        Mario.__init__(self, 1, 1)

class Mario12(Mario):
    def __init__(self):
        Mario.__init__(self, 1, 2)

class Mario13(Mario):
    def __init__(self):
        Mario.__init__(self, 1, 3)

class Mario14(Mario):
    def __init__(self):
        Mario.__init__(self, 1, 4)

class Mario21(Mario):
    def __init__(self):
        Mario.__init__(self, 2, 1)

class Mario22(Mario):
    def __init__(self):
        Mario.__init__(self, 2, 2)

class Mario23(Mario):
    def __init__(self):
        Mario.__init__(self, 2, 3)

class Mario24(Mario):
    def __init__(self):
        Mario.__init__(self, 2, 4)
class Mario31(Mario):
    def __init__(self):
        Mario.__init__(self, 3, 1)

class Mario32(Mario):
    def __init__(self):
        Mario.__init__(self, 3, 2)

class Mario33(Mario):
    def __init__(self):
        Mario.__init__(self, 3, 3)

class Mario34(Mario):
    def __init__(self):
        Mario.__init__(self, 3, 4)

class Mario41(Mario):
    def __init__(self):
        Mario.__init__(self, 4, 1)

class Mario42(Mario):
    def __init__(self):
        Mario.__init__(self, 4, 2)

class Mario43(Mario):
    def __init__(self):
        Mario.__init__(self, 4, 3)

class Mario44(Mario):
    def __init__(self):
        Mario.__init__(self, 4, 4)

class Mario51(Mario):
    def __init__(self):
        Mario.__init__(self, 5, 1)

class Mario52(Mario):
    def __init__(self):
        Mario.__init__(self, 5, 2)

class Mario53(Mario):
    def __init__(self):
        Mario.__init__(self, 5, 3)

class Mario54(Mario):
    def __init__(self):
        Mario.__init__(self, 5, 4)


class Mario61(Mario):
    def __init__(self):
        Mario.__init__(self, 6, 1)

class Mario62(Mario):
    def __init__(self):
        Mario.__init__(self, 6, 2)

class Mario63(Mario):
    def __init__(self):
        Mario.__init__(self, 6, 3)

class Mario64(Mario):
    def __init__(self):
        Mario.__init__(self, 6, 4)

class Mario71(Mario):
    def __init__(self):
        Mario.__init__(self, 7, 1)

class Mario72(Mario):
    def __init__(self):
        Mario.__init__(self, 7, 2)

class Mario73(Mario):
    def __init__(self):
        Mario.__init__(self, 7, 3)

class Mario74(Mario):
    def __init__(self):
        Mario.__init__(self, 7, 4)


class Mario81(Mario):
    def __init__(self):
        Mario.__init__(self, 8, 1)

class Mario82(Mario):
    def __init__(self):
        Mario.__init__(self, 8, 2)

class Mario83(Mario):
    def __init__(self):
        Mario.__init__(self, 8, 3)

class Mario84(Mario):
    def __init__(self):
        Mario.__init__(self, 8, 4)
