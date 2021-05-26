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


# Reload gym-tetris env for 10 times.(prevent picture chaos)
class Tetris2CustomState(gym.Wrapper):
    def __init__(self):
        self.grid_height = 23
        self.grid_width = 10
        
        self.observation_space = spaces.Box(low=0, high=10, shape=(5, self.grid_height, self.grid_width), dtype=np.float32)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = process_frame(state, info)
        return state, reward, done, info


    def reset(self):
        return process_frame(self.env.reset(), {})

    # process frame
    def process_frame(self, action, info):
        # grid_ori    : 1:blank, 4: control, 2: block  => 1: blank, 4: block, 6: control
        # grid_ori_detail :  1: blank, 2:line blank, 3: half blank, 4: block, 5: dead blank
        # grid_drop    : 1: blank, 4: block, 6: control
        # grid_drop_detail :  1: blank, 2:line blank, 3: half blank, 4: block, 5: dead blank
        # grid_info  : 0 none, 7 : info, (line<=6)   8 : score up/ can clear, 9: score down

        # init grid_ori
        grid_ori = state[6:, :]
        grid_ori = grid_ori * 2
        grid_ori = grid_ori.astype(np.uint8)

        # init grid_info
        grid_info = state[:6, :]
        grid_info = grid_info * 7
        grid_info = grid_info.astype(np.uint8)
        grid_info = np.append(grid_info, np.zeros((self.grid_height-6,10)), axis=0)

        #grid_ori    : 1:blank, 4: control, 2: block  => 1: blank, 4: block, 6: control
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if grid_ori[i, j] == 4:
                    grid_ori[i, j] = 6
                elif grid_ori[i, j] == 2:
                    grid_ori[i, j] = 4
        
        # grid_ori_detail :  1: blank, 2:line blank, 3: half blank, 4: block, 5: dead blank
        grid_ori_detail = self.mark_detail(grid_ori, False)
        
        # computer now score
        ori_grid_score, ori_clear_line_num, ori_line_blank, ori_half_holes, ori_dead_holes, ori_border_height = self.computer_detail(grid_ori_detail)
        
        # grid_drop    : 1: blank, 4: block, 6: control
        grid_drop = self.drop_down(grid_ori)
        
        # grid_drop_detail :  1: blank, 2:line blank, 3: half blank, 4: block, 5: dead blank
        grid_drop_detail = self.mark_detail(grid_drop, True)
        
        drop_grid_score, drop_clear_line_num, drop_line_blank, drop_half_holes, drop_dead_holes, drop_border_height = self.computer_detail(grid_drop_detail)
        
        # grid_info  : 0 none, 7 : info, (line<=6)   8 : score up/ can clear, 9: score down
        pos = self.grid_height - 1
        if drop_grid_score >= ori_grid_score: 
            grid_info[pos, 0] = 8
        else:
            grid_info[pos, 0] = 9

        if drop_clear_line_num > 0: 
            grid_info[pos, 1] = 8

        if drop_line_blank > ori_line_blank: 
            grid_info[pos, 2] = 9
        else:
            grid_info[pos, 2] = 8

        if drop_half_holes > ori_half_holes: 
            grid_info[pos, 3] = 9
        else:
            grid_info[pos, 3] = 8

        if drop_dead_holes > ori_dead_holes: 
            grid_info[pos, 4] = 9
        else:
            grid_info[pos, 4] = 8

        if drop_border_height > ori_border_height: 
            grid_info[pos, 5] = 9
        else:
            grid_info[pos, 5] = 8

        info['ori_score'] = [ori_grid_score, ori_clear_line_num, ori_line_blank, ori_half_holes, ori_dead_holes, ori_border_height]
        info['drop_score'] = [drop_grid_score, drop_clear_line_num, drop_line_blank, drop_half_holes, drop_dead_holes, drop_border_height]

        #return state
        np.array((grid_ori, grid_ori_detail, grid_drop, grid_drop_detail, grid_info))

            
    # grid_drop    : 1: blank, 4: block, 6: control
    def drop_down(self, ori_grid):
        drop_grid = ori_grid.copy()
        drop_end = False
        find_ctrl_obj = False
        
        for i in reversed(range(self.grid_height)):
            if drop_end:
                break

            for j in range(self.grid_width):
                if ori_grid[i, j] == 6:
                    find_ctrl_obj = True

                    if i >= grid_height-1:
                        drop_end = True
                        break

                    elif ori_grid[i+1, j] == 4:
                        drop_end = True
                        break

                    else:
                        drop_grid[i+1, j] = 5
                        drop_grid[i, j] = 1

        if not find_ctrl_obj:
            return drop_grid
        
        if drop_end:
            return ori_grid
        else:
            return self.drop_down(drop_grid):
        

    # search grid
    # grid_ori    : 1: blank, 4: block, 6: control
    # grid_ori_detail :  1: blank, 2:line blank, 3: half blank, 4: block, 5: dead blank, 6: control
    def search_noblock(self, ori_grid, detail_grid, posheight, poswidth, ctrl_block):
        grid_height = len(ori_grid)
        grid_width = len(ori_grid[0])

        if posheight<0 or poswidth<0 or posheight >= grid_height or poswidth >= grid_width:
            return

        if £¨not ctrl_block and ori_grid[posheight, poswidth] == 4£© or (ctrl_block and (ori_grid[posheight, poswidth] == 4 or ori_grid[posheight, poswidth] == 6)) :
            detail_grid[posheight, poswidth] = 4      #blocked
            return

        if detail_grid[posheight, poswidth] == 4:  # has set blocked
            return

        if detail_grid[posheight, poswidth] == 1:  # has set blank
            return

        if (not ctrl_block and ori_grid[posheight, poswidth] != 4) or (ctrl_block and (ori_grid[posheight, poswidth] != 4 and ori_grid[posheight, poswidth] != 6)):
            detail_grid[posheight, poswidth] = 1     # blank

            self.search_noblock(ori_grid, detail_grid, posheight, poswidth-1, ctrl_block)
            self.search_noblock(ori_grid, detail_grid, posheight, poswidth+1, ctrl_block)
            self.search_noblock(ori_grid, detail_grid, posheight-1, poswidth, ctrl_block)
            self.search_noblock(ori_grid, detail_grid, posheight+1, poswidth, ctrl_block)


    #detail :  1: blank, 2:line blank, 3: half blank, 4: block, 5: dead blank
    def mark_detail(self, ori_grid, ctrl_block):
        detail_grid = np.zeros((self.grid_height, self.grid_width))
        
        # grid_detail: 0: not set, 1: blank, 4: block
        self.search_noblock(ori_grid, detail_grid, 0, 0, ctrl_block)
        
        # grid_detail :  1: blank, 2:line blank, 3: half blank, 4: block, 5: dead blank, 6: control
        for j in range(self.grid_width):
            has_blocked = False
            for i in range(self.grid_height):
                if £¨not ctrl_block and ori_grid[posheight, poswidth] == 4£© or (ctrl_block and (ori_grid[posheight, poswidth] == 4 or ori_grid[posheight, poswidth] == 6)) :
                    has_blocked = True

                if detail_grid[i, j] == 0:
                    if £¨not ctrl_block and ori_grid[posheight, poswidth] == 4£© or (ctrl_block and (ori_grid[posheight, poswidth] == 4 or ori_grid[posheight, poswidth] == 6)) :
                        detail_grid[i, j] = 4          # blocked  4
                    else:
                        detail_grid[i, j] = 5         # dead hole  5
                elif detail_grid[i, j] == 1:
                    if has_blocked:
                        detail_grid[i, j] = 3          # half blank  3
                    else:
                        detail_grid[i, j] == 1         # blank  1
                elif detail_grid[i, j] == 4:
                    detail_grid[i, j] = 4             # blocked  4
        
        # grid_detail :  2:line blank, 6: control
        for i in range(self.grid_height):
            has_block = False
            for j in range(self.grid_width):
                if detail_grid[i, j] == 4:
                    has_block = True
                    break

            for j in range(grid_width):
                if detail_grid[i, j] == 1:             # ori blank 1
                    if has_block:
                        detail_grid[i, j] = 2            # line blank 2
        
        return detail_grid



    #detail :  1: blank, 2:line blank, 3: half blank, 4: block, 5: dead blank
    def computer_detail(self, detail_grid):
        grid_score = 0
        clear_line_num = 0
        line_blank = 0
        half_holes = 0
        dead_holes = 0
        border_height = 0
        line_all_blank_height = 0
        
        for i in range(self.grid_height):
            line_can_clear = True
            line_all_blank = True
            
            for j in range(self.grid_width):
                if detail_grid[i, j] != 1:
                    line_all_blank = False
                    break
            
            if line_all_blank:
                line_all_blank_height += 1

            for j in range(self.grid_width):
                if detail_grid[i, j] != 4:
                    line_can_clear = False
                    break
                    
            if line_can_clear:
                clear_line_num += 1
                
            
            for j in range(self.grid_width):
                if detail_grid[i, j] == 1:
                    grid_score += 0
                elif detail_grid[i, j] == 2:
                    line_blank += 1
                    grid_score -= 2
                elif detail_grid[i, j] == 3:
                    half_holes += 1
                    grid_score -= 3
                elif detail_grid[i, j] == 4:
                    grid_score += 4
                elif detail_grid[i, j] == 5:
                    dead_holes += 1
                    grid_score -= 5
                    
        grid_score += 40 * pow(clear_line_num, 4)
        border_height = sellf.grid_height - line_all_blank_height
        
        return grid_score, clear_line_num, line_blank, half_holes, dead_holes, border_height

# Reload gym-tetris env for 10 times.(prevent picture chaos)
class Tetris2Movedown(gym.Wrapper):
    def __init__(self):
        super(Tetris2Movedown, self).__init__(env)
        self.curr_frame = 0

    def step(self, action):
        self.curr_frame += 1
        
        state, reward, done, info = self.env.step(action)
        last_state = state
        last_reward = reward
        last_done = done
        last_info = info


        # every 2 frame soft drop once
        if self.curr_frame % 2  and not last_done:
            state2, reward2, done2, info2 = self.env.step(4)       #soft drop
            last_state = state2
            last_reward += reward2
            last_done = done2
            last_info = info2

        # every 10 frame soft drop twice to prevent loop
        if self.curr_frame % 5  and not last_done:
            state3, reward3, done3, info3 = self.env.step(4)       #soft drop
            last_state = state3
            last_reward += reward3
            last_done = done3
            last_info = info3

        return last_state, last_reward, last_done, last_info


    def reset(self):
        self.curr_frame = 0
        
        return self.env.reset()


# The last wrapped env
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
        self.env = Tetris2Movedown(self.env)
        self.env = Tetris2CustomState(self.env)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.last_state = None
        self.last_reward = 0
        self.last_done = False
        self.last_info = None

        self.max_reward = -100
        self.curr_reward_sum = 0



    def step(self, action):
        ##if self.last_done:
        #    self.reset()

        self.curr_frame += 1

        state, reward, done, info = self.env.step(action)
        self.last_state = state
        self.last_reward = reward
        self.last_done = done
        self.last_info = info

        # reward sum
        self.curr_reward_sum += self.last_reward


        if self.curr_reward_sum > self.max_reward:
            self.max_reward = self.curr_reward_sum
            print(">> MAX REWARD : reward:{}".format(self.max_reward))

        #self.render()
        return self.last_state, self.last_reward, self.last_done, self.last_info

    def reset(self):
        self.last_state = self.env.reset()
        self.last_reward = 0
        self.last_done = False
        self.last_info = None

        self.curr_reward_sum = 0

        return self.last_state

    def render(self, mode='human'):
        # if self.last_reward != 0:
        if True:
            print("-------------------------------------------------------------")
            print(self.last_state)
            print("Reward  Total:{},  Last:{}".format(self.curr_reward_sum, self.last_reward))
            print("Done:{}".format(self.last_done))
            print("Info:{}".format(self.last_info))
            time.sleep(0.3)
        else:
            return

    def close (self):
        return self.env.close()

    def print_status(self):
        return print(self.env)
