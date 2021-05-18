import os
import gym
import numpy as np
import cv2
from gym import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym.spaces import Box
import random
from PIL import Image, ImageDraw



class RecorderVideo(gym.Wrapper):
    def __init__(self, env, saved_path, min_record_length = 20):
        super(RecorderVideo, self).__init__(env)
        self.has_recorded = False
        self.record_length = 0
        self.min_record_length = min_record_length
        self.record_height = self.observation_space.shape[0]
        self.record_width = self.observation_space.shape[1]
        self.saved_path = saved_path

        self.record_video = []


    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)
        return self.record(state), reward, done, info

    def reset(self):
        if not self.has_recorded and self.record_length > self.min_record_length:
            self.has_recorded = True
            print("Recoard video to {}, total frame is {}".format(self.saved_path, self.record_length))

            if os.path.exists(self.saved_path):
                os.remove(self.saved_path)

            self.record_video[0].save(self.saved_path, save_all=True, append_images=self.record_video[1:], optimize=True, duration=20, loop=0)

        elif not self.has_recorded:
            self.record_length = 0
            print("Recoard frame is {} (< {}), continue recording!".format(self.record_length, self.min_record_length))

        return self.record(self.env.reset())


    def record(self, image_array):
        if not self.has_recorded:
            img = Image.fromarray(image_array, 'RGB')
            self.record_video.append(img)
            self.record_length += 1

        return image_array


class CustomSkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.skip = skip

    def step(self, action) -> GymStepReturn:
        last_state = None
        total_reward = 0
        last_done = False
        last_info = None

        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            last_state = state
            last_done = done
            last_info = info
            if done:
                return state, total_reward, done, info

        return last_state, total_reward, last_done, last_info


class RandomStart(gym.Wrapper):
    def __init__(self, env, rnum=5):
        super(RandomStart, self).__init__(env)
        self.rnum = rnum
        self.is_start = True

    def step(self, action) -> GymStepReturn:

        if self.is_start:
            self.is_start = False

            last_state = None
            total_reward = 0
            last_done = False
            last_info = None

            for i in range(random.randint(1, self.rnum*2+1)):
                state, reward, done, info = self.env.step(self.env.action_space.sample())
                total_reward += reward
                last_state = state
                last_done = done
                last_info = info
                if done:
                    return state, total_reward, done, info

            state, reward, done, info = self.env.step(action)
            total_reward += reward
            last_state = state
            last_done = done
            last_info = info

            return last_state, total_reward, last_done, last_info
        else:
            return self.env.step(action)


    def reset(self):
        state = self.env.reset()
        self.env.seed()
        random.seed()
        self.is_start = True
        for i in range(random.randint(1, self.rnum*2+1)):
            for j in range(random.randint(1, self.rnum*2+1)):
                self.env.step(self.env.action_space.sample())
            self.env.reset()

        return self.env.reset()
