import gym
import numpy as np
import cv2
from gym import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym.spaces import Box
import random
import subprocess as sp


class RecorderVideo(gym.Wrapper):
    def __init__(self, env, saved_path, min_record_length = 20):
        super(RecorderVideo, self).__init__(env)
        self.has_recorded = False
        self.record_length = 0
        self.min_record_length = min_record_length
        self.record_height = self.observation_space.shape[0]
        self.record_width = self.observation_space.shape[1]
        self.saved_path = saved_path
        

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(self.record_width, self.record_height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)
        return self.record(state), reward, done, info

    def reset(self):
        if not self.has_recorded and self.record_length > self.min_record_length:
            self.has_recorded = True
            print("Recoard video to {}, total frame is {}".format(self.saved_path, self.record_length))
            try:
                self.pipe.terminate()        
            except:
                pass
        elif not self.has_recorded:
            self.record_length = 0
            print("Recoard frame is {} (< {}), continue recording!".format(self.record_length, self.min_record_length))
        
        return self.process_frame(self.env.reset())
        
        
    def record(self, image_array):
        if not self.has_recorded: 
            self.pipe.stdin.write(image_array.tostring())
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
