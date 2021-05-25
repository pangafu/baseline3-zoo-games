#from __future__ import print_function



# This File is Originally taken from https://github.com/jaybutera/tetrisRL. 
# There are major modification to this environment to be fine with the model chosen.

import numpy as np
import random
from controllers import basic_evaluation_fn,best_action
from utils.tetris_engine_utils import *


class TetrisEngine:

    def __init__(self, width, height ,
    state_calculator='schwenker2008',eval_fn='dellacherie', penalty =-100,
    clamp_diff=3,melax_factor=2):
    
        self.width = int(width)
        self.height = int(height)
        self.state_calculator = state_calculator
        self.eval_fn = eval_fn
        self.melax_factor = melax_factor
        self.clamp_diff = clamp_diff
        self.penalty = penalty
        self.board = np.zeros(shape=(width, height), dtype=np.float)

        # actions are triggered by letters
        self.value_action_map = {
            0: left,
            1: right,
            2: hard_drop,
            3: soft_drop,
            4: rotate_left,
            5: rotate_right,
            6: idle,
        }
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)
        
        
        
        self.group_actions_number=4*self.width # Means that there are 4 possible rotations and width-number of translations +1 for idle translations
        
        # for running the engine
        self.time = -1
        self.score = -1
        self.anchor = None
        self.shape = None
        
        self.prev_state_evaluation=0
        self.landing_height=0
        self.cleared_lines=0
        self.cleared_lines_per_move=0
        # used for generating shapes
        self._shape_counts = [0] * len(shapes)
        
        self.piece_number=None
        self.tetrominos=0
        self.total_reward=0
        # clear after initializing
        self.clear()
    
    
    
    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                self.piece_number=i
                return shapes[shape_names[i]]

    def _new_piece(self):
        # Place randomly on x-axis with 2 tiles padding
        #x = int((self.width/2+1) * np.random.rand(1,1)[0,0]) + 2
        self.tetrominos+=1
        self.anchor = (self.width //2, 0)
        #self.anchor = (x, 0)
        self.shape = self._choose_shape()
        self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)

        
    # Modification
    
    
    def _has_dropped(self):
        is_occ,self.landing_height=is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board,h=True)
        return is_occ
    
    
    #Modification
    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        self.score += sum(can_clear)
        self.board = new_board
        self.cleared_lines_per_move=sum(can_clear)
        self.cleared_lines += sum(can_clear)


    #Modification
    
    #Modification
    
    def _rotate_grouped_action(self,rotations):
        action_taken =5 if rotations==1 else 4
        
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action_taken](self.shape, self.anchor, self.board)
        
        if rotations==3:# face up so that means 2 rotate_left or 2 rotate_right
            self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
            self.shape, self.anchor = self.value_action_map[action_taken](self.shape, self.anchor, self.board)
    
    def _translate_grouped_action(self,translations,is_right):
        action_taken=1 if is_right else 0
        for _ in range(translations):
            self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
            self.shape, self.anchor = self.value_action_map[action_taken](self.shape, self.anchor, self.board)
    
    
    def _exec_grouped_actions(self,action):
        
        
        #Basic Actions
        #0: left,
        #1: right,
        #2: hard_drop,
        #3: soft_drop,
        #4: rotate_left,
        #5: rotate_right,
        #6: idle,
        
        #0->10  idle, for all v in first five values means move v+1 to the right and the last 5 moves means v+1-5 to the left
        # the same for the other grouped actions 
        #it is stated that it is 0->10 because the expected width is 10 
        is_right=False
        rotations =action//(self.width)
        translations =action%(self.width)
        if translations>=self.width//2:
            is_right=True
            translations-=self.width//2
        else:
            translations=self.width//2 - translations
        
        if rotations>0:
            self._rotate_grouped_action(rotations)
        if translations>0:    
            self._translate_grouped_action(translations,is_right)
        
        self.shape, self.anchor = hard_drop(self.shape, self.anchor, self.board)

    
   
    def calc_state(self):
        

        state=basic_evaluation_fn(self,self.state_calculator,abs_value=False,melax_factor=self.melax_factor,clamp_diff=self.clamp_diff)
        state=np.append(state,self.piece_number)
        
        return state
    
    
    def calc_reward(self):
        
        state_evaluation= self.calc_state_evaluation()
        # tm=self.time
        # if tm>5000:
        #     tm=5000
        reward=state_evaluation-self.prev_state_evaluation +self.time
        self.prev_state_evaluation=state_evaluation
        return reward
    
    def _exec_normal_action(self,action):
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)
        self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)


    def step(self, action):
        
        #Basic Actions for normal actions but the groud are up tp 40 actions
        #0: left,
        #1: right,
        #2: hard_drop,
        #3: soft_drop,
        #4: rotate_left,
        #5: rotate_right,
        #6: idle,
        
        # if self.grouped_actions:
        self._exec_grouped_actions(action)
        # else:
        #     self._exec_normal_action(action)
        # Update time and reward
        self.time += 1
        reward=0


        done = False
        if self._has_dropped():
            
            self._set_piece(True)
            self._clear_lines()
            state = self.calc_state()
            if np.any(self.board[:, 0]):
                done = True
            else:
                self._new_piece()
            self._set_piece(False)

        
        #calcualte the Reward based on the Evaluation of the states. 
        
        
        reward = self.penalty if done else self.calc_reward() 
        self.total_reward+=reward
        return state, round(reward,3), done

    def clear(self):
        self.time = 0
        self.score = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)
        self.prev_state_evaluation=0
        self.landing_height=0
        self.cleared_lines=0
        self.cleared_lines_per_move=0
        self.tetrominos=0
        self.total_reward=0
        return self.calc_state()

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on

    def __repr__(self):
        self._set_piece(True)
        s = 'o' + '-' * self.width + 'o\n'
        s += '\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]) + '|' for i in self.board.T])
        s += '\no' + '-' * self.width + 'o'
        self._set_piece(False)
        return s
    
    
    # ADDED Functions
    #Modification   
    def random_action(self):
        return int(np.random.random()*self.group_actions_number)
    
    def number_actions(self):
        return self.group_actions_number
    
    def state_shape(self):
        return np.prod(self.calc_state().shape)
    
    
    
    def calc_state_evaluation(self):
        
        return basic_evaluation_fn(self,self.eval_fn,melax_factor=self.melax_factor,clamp_diff=self.clamp_diff)
        
                
if __name__=="__main__":
    engine = TetrisEngine(10,20)
    done = False 
    while not done: 
        _,_,done=engine.step(1)
        print(engine)





