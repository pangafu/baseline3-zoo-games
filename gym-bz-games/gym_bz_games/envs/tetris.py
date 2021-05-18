import gym
import numpy as np
import cv2
from gym import spaces
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym_bz_games.wrappers import CustomSkipFrame, NesFrameGray, NesFrameGrayHalf, NesFrameGrayCrop, RandomStart, NesFrameBinary, NesFrameGrayScale, RecorderVideo
from gym_tetris.actions import MOVEMENT
import gym_tetris



def create_env(self):
    env = gym_tetris.make("TetrisA-v0")
    env = JoypadSpace(env, MOVEMENT)
    return env

# Reload gym-tetris env for 10 times.(prevent picture chaos)
class TetrisEnvReload(gym.Wrapper):
    def __init__(self, env, reload_resettime=10):
        super(TetrisReload, self).__init__(env)
        self.reload_resettime = reload_resettime
        self.reset_time = 0

    def reset(self):
        self.reset_time += 1

        if self.reset_time > self.reload_resettime:
            print("Reload gym-tetris env for {} times to prevent picture chaos".format(self.reload_resettime))
            self.env.close()
            self.env = create_env()
            self.reset_time = 0

        return self.env.reset()


class CustomReward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.create_env()
        self.max_reward = 0
        self.max_score = 0
        self.curr_score = 0
        self.curr_lines = 0
        self.curr_board = 0
        self.curr_totaluse = 0
        self.curr_reward_sum = 0
        self.curr_step_num = 0
        self.ingore_use_reward = False

    def step(self, action: int) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)

        reward = 0.
        #reward += (info["score"] - self.curr_score)/20.
        self.curr_score = info["score"]

        reward += (info["number_of_lines"] - self.curr_lines) * 10.
        self.curr_lines = info["number_of_lines"]

        if info["board_height"] > self.curr_board and self.curr_totaluse>=2:
            self.ingore_use_reward = True

        if info["board_height"] > 6 or  info["board_height"] < self.curr_board:
            reward -= 5. * (info["board_height"] - self.curr_board)

        self.curr_board = info["board_height"]

        totaluse = 0
        for value in info["statistics"].values():
            totaluse += value

        self.curr_step_num += 1
        #if self.curr_step_num % 80 == 1:
        #    reward -= 1


        if self.curr_board <= 20:
            if totaluse > self.curr_totaluse:
                 if self.ingore_use_reward:
                     reward -= 1
                     self.ingore_use_reward = False
                 else:
                     reward += 2

        self.curr_totaluse = totaluse


        #if reward < -1*self.curr_reward_sum:
        #    reward = -1*self.curr_reward_sum

        if self.curr_board - self.curr_lines - self.curr_totaluse/5. > 8:
            done = True

        if self.curr_board > 15:
            done = True

        if done:
            reward -= max(15 - self.curr_lines - self.curr_totaluse/5, 0)

        self.curr_reward_sum += reward

        #if reward != 0:
        #    print(reward)

        if self.curr_reward_sum > self.max_reward:
            self.max_reward = self.curr_reward_sum
            print(">> MAX REWARD : reward:{}  score:{}  lines:{}  board:{}  totaluse:{}".format(self.max_reward, self.curr_score, self.curr_lines, self.curr_board, self.curr_totaluse))


        return state, reward, done, info


    def reset(self, **kwargs) -> GymObs:
        self.curr_score = 0
        self.curr_lines = 0
        self.curr_board = 0
        self.curr_totaluse = 0
        self.curr_reward_sum = 0
        self.curr_step_num = 0
        self.ingore_use_reward = False

        return self.env.reset(**kwargs)


    def process_frame(self, frame):
        
        
        return frame


class Tetris(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        env = create_env()
        
        need_record = False

        bz_record = os.environ.get('BZ_RECORD')
        bz_record_algo = os.environ.get('BZ_RECORD_ALGO')

        if bz_record and bz_record == "1":
            need_record = True


        env = TetrisEnvReload(env, reload_resettime=10)
        env = CustomSkipFrame(env, skip = 8)

        if need_record:
            env = RecorderVideo(env, saved_path=os.path.join("videoes", bz_record_algo, "SuperMarioBros-{}-{}-v0.gif".format(self.world, self.stage)))

        env = NesFrameGrayHalf(env)
        env = NesFrameGrayCrop(env)
        env = NesFrameGrayScale(env, scale=0.25)
        env = NesFrameBinary(env)
        env = RandomStart(env, rnum = 2)

        env = CustomReward(env)

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


