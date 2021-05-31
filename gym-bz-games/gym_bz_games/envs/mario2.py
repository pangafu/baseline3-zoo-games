import os
import gym
import numpy as np
import cv2
import random
from gym import spaces
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym_bz_games.wrappers import CustomSkipFrame, NesFrameGray, NesFrameGrayHalf, RandomStart, RecorderVideoTools
import gym_super_mario_bros


# actions for very simple movement
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['down'],
]


class RecorderVideo(gym.Wrapper):
    def __init__(self, env, saved_path, min_record_length = 20):
        super(RecorderVideo, self).__init__(env)
        self.has_recorded = False
        self.min_record_length = min_record_length
        self.recorder = RecorderVideoTools(saved_path)
        self.record_done = False

    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)

        self.record_done = info["flag_get"]
        # self.record_done = info["x_pos"] > 2850
        if self.record_done:
           done = True
        return self.record(state), reward, done, info

    def reset(self):
        if not self.has_recorded:
            if self.recorder.record_length > self.min_record_length and self.record_done:
                self.has_recorded = True
                self.recorder.save()
            else:
                if self.recorder.record_length > self.min_record_length:
                    print("Record frame is {} ( Min Length {}), Record Done is {}, continue recording!".format(self.recorder.record_length, self.min_record_length, self.record_done))
                self.recorder.reset()

        self.last_done = False
        return self.record(self.env.reset())


    def record(self, image_array):
        if not self.has_recorded:
            self.recorder.record(image_array)

        return image_array


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
        reward += (info["score"] - self.curr_score) / 50.
        self.curr_score = info["score"]

        #reward += (info["coins"] - self.curr_coins) * 20.
        self.curr_coins = info["coins"]

        player_status = 0
        if info["status"] != "small" and info["status"] != "tall":
            print(info["status"])


        if info["status"] == "tall":
            player_status = 1

        if info["status"] != "small" and info["status"] != "tall":
            player_status = 2

        reward += (player_status - self.curr_status) * 10.
        self.curr_status = player_status

        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.curr_x - 500:
                reward -= 50
                done = True

        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
                reward = -50
                done = True


        self.curr_x = info["x_pos"]

        self.curr_frame += 1

        if self.curr_x > self.curr_x_max + 5:
            self.curr_x_max = self.curr_x_max + 5
            reward += 2
            self.curr_x_max_frame = self.curr_frame


        if self.curr_frame - self.curr_x_max_frame > 700:
            done = True



        if done:
            if info["flag_get"]:
                reward += 200
                print(">> FLAG GET!!! : world:{} stage:{} reward:{}  score:{}  coin:{}  status:{}  x:{}".format(self.world, self.stage, self.max_reward, self.curr_score, self.curr_coins, self.curr_status, self.curr_x))
            else:
                reward -= 20

        self.curr_reward_sum += reward

        if self.curr_reward_sum > self.max_reward:
            self.max_reward = self.curr_reward_sum
            if self.max_reward % 20 == 0:
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




class MarioRandom(gym.Env):
    metadata = {'render.modes':['human']}

    def make_env(self):
        if self.env is not None:
            self.env.close()

        self.world = random.randint(1, 8)
        self.stage = random.randint(1,4)

        if self.need_record:
            print(">> ENV: world:{}  stage:{}".format(self.world, self.stage))

        env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(self.world, self.stage))
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        env = CustomSkipFrame(env, skip = 2)

        if self.need_record:
            env = RecorderVideo(env, saved_path=os.path.join("videoes", self.bz_record_algo, "SuperMarioBros-Random-{}-{}-v0.gif".format(self.world, self.stage)))

        env = NesFrameGrayHalf(env)

        env = CustomReward(env, self.world, self.stage)


        self.env = env

        self.action_space = env.action_space
        self.observation_space = env.observation_space


    def __init__(self):
        self.need_record = False

        bz_record = os.environ.get('BZ_RECORD')
        self.bz_record_algo = os.environ.get('BZ_RECORD_ALGO')

        if bz_record and bz_record == "1":
            self.need_record = True

        self.env = None

        self.make_env()



    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.make_env()
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

    def close (self):
        return self.env.close()



