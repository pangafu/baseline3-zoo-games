import os
import gym
import numpy as np
import cv2
from gym import spaces
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from gym_bz_games.wrappers import CustomSkipFrame, NesFrameGray, NesFrameGrayHalf, NesFrameGrayCrop, RandomStart, NesFrameBinary, NesFrameGrayScale, RecorderVideo
from gym_tetris.actions import MOVEMENT
import gym_tetris
from gym.spaces import Box


def create_env():
    env = gym_tetris.make("TetrisA-v0")
    env = JoypadSpace(env, MOVEMENT)
    return env


# Reload gym-tetris env for 10 times.(prevent picture chaos)
class TetrisEnvReload(gym.Wrapper):
    def __init__(self, env, reload_resettime=10):
        super(TetrisEnvReload, self).__init__(env)
        self.reload_resettime = reload_resettime
        self.reset_time = 0

    def reset(self):
        self.reset_time += 1

        if self.reset_time > self.reload_resettime:
            #print("Reload gym-tetris env for {} times to prevent glitch".format(self.reload_resettime))
            self.env.close()
            self.env = create_env()
            self.reset_time = 0

        return self.env.reset()


class CustomReward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.max_reward = 0
        self.max_score = 0
        self.curr_score = 0
        self.curr_lines = 0
        self.curr_board = 0
        self.curr_totaluse = 0
        self.curr_reward_sum = 0
        self.curr_step_num = 0

        self.shape_height = self.observation_space.shape[0]
        self.shape_width = self.observation_space.shape[1]
        self.stack_grid = np.zeros((self.shape_height,self.shape_width))
        self.upper_grid = np.zeros((self.shape_height,self.shape_width))

        self.curr_dead_holes = 0
        self.curr_half_holes = 0
        self.curr_line_blank = 0
        self.curr_grid_score = 0

        self.info_dead_holes = 0
        self.info_half_holes = 0
        self.info_line_blank = 0
        self.info_grid_score = 0

        self.observation_space = Box(low=0, high=5, shape=(2, self.shape_height, self.shape_width), dtype=np.uint8)

    def step(self, action: int) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)
        state = self.process_frame(state)

        reward = 0.

        #reward += (info["score"] - self.curr_score)/20.
        self.curr_score = info["score"]

        self.curr_board = info["board_height"]

        totaluse = 0
        for value in info["statistics"].values():
            totaluse += value

        self.curr_totaluse = totaluse

        self.curr_step_num += 1

        self.curr_dead_holes = self.info_dead_holes
        self.curr_half_holes = self.info_half_holes
        self.curr_line_blank = self.info_line_blank


        reward += self.info_grid_score - self.curr_grid_score
        self.curr_grid_score = self.info_grid_score


        if self.curr_board >= 15:
            done = True

        if done:
            reward -= self.curr_half_holes*2 + self.curr_line_blank*3

        if info["number_of_lines"] != self.curr_lines:
            reward += pow((info["number_of_lines"] - self.curr_lines), 1.5) * 40.
            self.curr_lines = info["number_of_lines"]
            print(">> CLEAR LINE : reward:{} score:{} lines:{} board:{} blank:{} half:{} holes:{} total:{}".format(self.max_reward, self.curr_score,self.curr_lines, self.curr_board, self.curr_line_blank, self.curr_half_holes, self.curr_dead_holes, self.curr_totaluse))


        # reward sum
        self.curr_reward_sum += reward

        if self.curr_reward_sum > self.max_reward:
            self.max_reward = self.curr_reward_sum
            print(">> MAX REWARD : reward:{} score:{} lines:{} board:{} blank:{} half:{} holes:{} total:{}".format(self.max_reward, self.curr_score,self.curr_lines, self.curr_board, self.curr_line_blank, self.curr_half_holes, self.curr_dead_holes, self.curr_totaluse))


        return state, reward, done, info


    def reset(self, **kwargs) -> GymObs:
        self.curr_score = 0
        self.curr_lines = 0
        self.curr_board = 0
        self.curr_totaluse = 0
        self.curr_reward_sum = 0
        self.curr_step_num = 0

        self.stack_grid = np.zeros((self.shape_height,self.shape_width))
        self.upper_grid = np.zeros((self.shape_height,self.shape_width))

        self.curr_dead_holes = 0
        self.curr_half_holes = 0
        self.curr_line_blank = 0
        self.curr_grid_score = 0

        self.info_dead_holes = 0
        self.info_half_holes = 0
        self.info_line_blank = 0
        self.info_grid_score = 0


        self.env.reset(**kwargs)
        return np.zeros((2, self.shape_height, self.shape_width))


    def process_frame(self, frame):
        mask_frame = np.squeeze(frame)

        mask_reset = False
        for j in range(self.shape_width):
            if mask_frame[0, j] == 1:
                mask_reset = True
                break

        #print(mask_frame)

        if mask_reset:
            #print("Reset Stack Frame!")
            #print(mask_frame)
            self.stack_grid = np.zeros((self.shape_height,self.shape_width))

            detect_control_obj = True
            for i in range(self.shape_height):
                line_all_zero = True

                for j in range(self.shape_width):
                    if mask_frame[i, j] == 1:
                        line_all_zero = False

                for j in range(self.shape_width):
                    if detect_control_obj:
                        if mask_frame[i, j] == 1:
                            self.stack_grid[i, j] = 1    # clean
                        else:
                            self.stack_grid[i, j] = 1
                    else:
                        if mask_frame[i, j] == 1:
                            self.stack_grid[i, j] = 5     # blocked
                        else:
                            self.stack_grid[i, j] = 1


                if line_all_zero:
                    detect_control_obj = False

            #print(self.stack_grid)


        # stack obj is ok
        # mark all upper obj
        self.upper_grid = np.zeros((self.shape_height,self.shape_width))
        def search_open(posheight, poswidth):
            if posheight<0 or poswidth<0 or posheight >= self.shape_height or poswidth >= self.shape_width:
                return

            if self.stack_grid[posheight, poswidth] == 5:
                self.upper_grid[posheight, poswidth] = 5      #blocked
                return

            if self.upper_grid[posheight, poswidth] == 5:  # has set blocked
                return

            if self.upper_grid[posheight, poswidth] == 1:  # has set blocked
                return

            if self.stack_grid[posheight, poswidth] == 1:
                self.upper_grid[posheight, poswidth] = 1     # clean

                search_open(posheight, poswidth-1)
                search_open(posheight, poswidth+1)
                search_open(posheight-1, poswidth)
                search_open(posheight+1, poswidth)

        search_open(0, 0)


        # mask all half upper obj
        for j in range(self.shape_width):
            has_blocked = False
            for i in range(self.shape_height):
                if self.stack_grid[i, j] == 5:
                    has_blocked = True

                if self.upper_grid[i, j] == 0:
                    if self.stack_grid[i, j] == 5:
                        self.upper_grid[i, j] = 5          # blocked  5
                    if self.stack_grid[i, j] == 1:
                        self.upper_grid[i, j] = 4          # dead hole  4
                elif self.upper_grid[i, j] == 1:
                    if has_blocked:
                        self.upper_grid[i, j] = 3          # half dead hole  3
                    else:
                        self.upper_grid[i, j] == 1         # clean  1
                elif self.upper_grid[i, j] == 5:
                    self.upper_grid[i, j] = 5             # blocked  5

                if self.stack_grid[i, j] == 1 and mask_frame[i, j] == 1:
                    self.upper_grid[i, j] = 2             # control 2

        # update mask_frame
        self.mask_frame = mask_frame

        dead_holes = 0
        half_holes = 0
        line_blank = 0
        grid_score = 0

        #origin:     blank 1, line_blank 1, half_hole 3, blocked 5, dead_hole 4, control 2
        #modify to:  blank 0, line_blank 1, half_hole 2, blocked 3, dead_hole 4, control 5
        for i in range(self.shape_height):
            has_block = False
            for j in range(self.shape_width):
                if self.upper_grid[i, j] == 5:
                    has_block = True
                    break


            for j in range(self.shape_width):
                # upper grid
                if self.upper_grid[i, j] == 1:             # ori blank 1
                    if has_block and i>=5:
                        line_blank += 1
                        self.upper_grid[i, j] = 1          # line blank 1
                        grid_score -= 1
                    else:
                        self.upper_grid[i, j] = 0          # blank 0


                elif self.upper_grid[i, j] == 2:           # ori control 2
                    self.upper_grid[i, j] = 5              # control 5


                elif self.upper_grid[i, j] == 3:           # ori half_hole 3
                    half_holes += 1
                    self.upper_grid[i, j] = 2              # half_hole 2
                    grid_score -= 2


                elif self.upper_grid[i, j] == 4:           # ori dead_hole 4
                    dead_holes += 1
                    self.upper_grid[i, j] = 4              # dead_hole 4
                    grid_score -= 4

                elif self.upper_grid[i, j] == 5:           # ori blocked 5
                    self.upper_grid[i, j] = 3              # blocked 3
                    grid_score += 3


                # mask frame
                if self.mask_frame[i, j] == 1:
                    self.mask_frame[i, j] = 3              # block 3

        self.info_dead_holes = dead_holes
        self.info_half_holes = half_holes
        self.info_grid_score = grid_score
        self.info_line_blank = line_blank



        return np.array((self.mask_frame, self.upper_grid))


    def print_status(self):
        print(self.upper_grid)
        #print(self.stack_grid)
        #print(self.mask_frame)
        print(">> STATUS : reward:{}  score:{}  lines:{}  board:{}  holes:{}  half:{}  totaluse:{}".format(self.max_reward, self.curr_score, self.curr_lines, self.curr_board, self.curr_dead_holes, self.curr_half_holes, self.curr_totaluse))



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

    def print_status(self):
        return self.env.print_status()
