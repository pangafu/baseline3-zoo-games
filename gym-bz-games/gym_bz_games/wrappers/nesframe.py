import gym
import numpy as np
import cv2
from gym import spaces
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym.spaces import Box
import random


class NesFrameGray(gym.Wrapper):
    def __init__(self, env):
        super(NesFrameGray, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(240, 256, 1), dtype=np.uint8)

    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)
        return self.process_frame(state), reward, done, info

    def reset(self):
        return self.process_frame(self.env.reset())

    def process_frame(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame[:, :, None]
            return frame
        else:
            return np.zeros((240, 256, 1))


class NesFrameGrayHalf(gym.Wrapper):
    def __init__(self, env):
        super(NesFrameGrayHalf, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(120, 128, 1), dtype=np.uint8)

    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)
        return self.process_frame(state), reward, done, info

    def reset(self):
        return self.process_frame(self.env.reset())

    def process_frame(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (128, 120))[:, :, None]
            #frame = frame[None, :, :] / 255.
            return frame
        else:
            return np.zeros((120, 128, 1))


class NesFrameGrayCrop(gym.Wrapper):
    def __init__(self, env, left=48, right=88, top=24, bottom=104):
        super(NesFrameGrayCrop, self).__init__(env)
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.observation_space = Box(low=0, high=255, shape=(bottom-top, right-left, 1), dtype=np.uint8)

    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)
        return self.process_frame(state), reward, done, info

    def reset(self):
        return self.process_frame(self.env.reset())

    def process_frame(self, frame):
        if frame is not None:
            crop = frame[self.top:self.bottom, self.left:self.right]
            return crop
        else:
            return np.zeros((self.bottom-self.top, self.right-self.left, 1))


class NesFrameBinary(gym.Wrapper):
    def __init__(self, env, binline = 5):
        super(NesFrameBinary, self).__init__(env)
        self.binline = binline

    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)
        return self.process_frame(state), reward, done, info

    def reset(self):
        return self.process_frame(self.env.reset())

    def process_frame(self, frame):
        if frame is not None:
            frame = (frame > self.binline).astype(np.uint8)
            frame[frame == 1] = 255
            return frame
            
        return frame

class NesFrameGrayScale(gym.Wrapper):
    def __init__(self, env, scale=0.5):
        super(NesFrameGrayScale, self).__init__(env)
        oriheight = self.observation_space.shape[0]
        oriwidth = self.observation_space.shape[1]
        self.newheight = round(oriheight * scale)
        self.newwidth = round(oriwidth * scale)
        self.observation_space = Box(low=0, high=255, shape=(self.newheight, self.newwidth, 1), dtype=np.uint8)

    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)
        return self.process_frame(state), reward, done, info

    def reset(self):
        return self.process_frame(self.env.reset())

    def process_frame(self, frame):
        if frame is not None:
            frame = cv2.resize(frame, (self.newwidth, self.newheight))[:, :, None]
            return frame
        else:
            return np.zeros((self.newheight, self.newwidth, 1))

