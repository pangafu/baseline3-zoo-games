import random
import numpy as np
import time
import copy

i_index = 1
j_index = 2
l_index = 3
o_index = 4
s_index = 5
t_index = 6
z_index = 7

# each mino's layout, 4(rotation) x 4 x 2(blocks occupying [x, y], 
# where x and y value are offset values from upper left corner position)
I_layout = [[[1,2], [2,2], [3,2], [4,2]],
            [[2,1], [2,2], [2,3], [2,4]],
            [[0,2], [1,2], [2,2], [3,2]],
            [[2,0], [2,1], [2,2], [2,3]]]
J_layout = [[[0,0], [0,1], [1,1], [2,1]],
            [[1,0], [2,0], [1,1], [1,2]],
            [[0,1], [1,1], [2,1], [2,2]],
            [[1,0], [1,1], [0,2], [1,2]]]
L_layout = [[[2,0], [0,1], [1,1], [2,1]],
            [[1,0], [1,1], [1,2], [2,2]],
            [[0,1], [1,1], [2,1], [0,2]],
            [[0,0], [1,0], [1,1], [1,2]]]
O_layout = [[[1,0], [2,0], [1,1], [2,1]],
            [[1,1], [2,1], [1,2], [2,2]],
            [[0,1], [1,1], [0,2], [1,2]],
            [[0,0], [1,0], [0,1], [1,1]]]
S_layout = [[[1,0], [2,0], [0,1], [1,1]],
            [[1,0], [1,1], [2,1], [2,2]],
            [[1,1], [2,1], [0,2], [1,2]],
            [[0,0], [0,1], [1,1], [1,2]]]
T_layout = [[[1,0], [0,1], [1,1], [2,1]],
            [[1,0], [1,1], [2,1], [1,2]],
            [[1,1], [2,1], [0,1], [1,2]],
            [[1,0], [0,1], [1,1], [1,2]]]
Z_layout = [[[1,0], [0,0], [1,1], [2,1]],
            [[2,0], [1,1], [2,1], [1,2]],
            [[1,1], [2,2], [0,1], [1,2]],
            [[1,0], [0,1], [1,1], [0,2]]]


# each mino's SRS information. For details, check https://tetris.wiki/Super_Rotation_System
i_srs = [[[0,0],[-1,0],[2,0],[-1,0],[2,0]],
         [[-1,0],[0,0],[0,0],[0,1],[0,-2]],
         [[-1,1],[1,1],[-2,1],[1,0],[-2,0]],
         [[0,1],[0,1],[0,1],[0,-1],[0,2]]] # 4 x 5 x 2
jlstz_srs = [[[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[1,0],[1,-1],[0,2],[1,2]],
            [[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]]] # 4 x 5 x 2
o_srs = [[[0,0]],
         [[0,-1]],
         [[-1,-1]],
         [[-1,0]]] # 4 x 1 x 2

# spawn location
i_spawn = [3, 0] #(x, y)
jlostz_spawn = [4, 1]

# ren bonux (index : num ren, elem : bonus)
# index : 0 ~ 20
ren_bonus = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
class tetrimino:
    # block_data is 4 x 4 x 2 list, each of 4 lists representing different rotation states
    # srs_data is 4 x 5 x 2 list needed to implement super rotation system (for detail, go to https://tetris.wiki/Super_Rotation_System)
    # current_rot {0: initial state, 1: 90 degree clockwise, 2: 180 clockwise(counterclockwise), 3: 90 degree counterclockwise}
    # spawn_pos is [x, y] position where the block initially appears
    def __init__(self, mino_id: int, block_data, srs_data, spawn_pos):
        self.mino_id = mino_id
        self.block_data = block_data
        self.srs_data = srs_data
        self.spawn_pos = spawn_pos
        self.current_rot = 0 
        self.current_pos = [spawn_pos[0], spawn_pos[1]]
    # Based on its current [x, y], current_rot, calculate the final position if it would be rotated, where angle is specified angle_offset
    # angle_offset = 1 or -1
    # return a list of [x, y]
    def srs_xy(self, angle_offset): 
        final_angle = (self.current_rot + angle_offset) % 4
        current_srs = self.srs_data[self.current_rot]
        final_srs = self.srs_data[final_angle]
        result = []
        for i in range(len(current_srs)): # i = 1 (o_mino) i = 5 (other minos)
            x = current_srs[i][0] - final_srs[i][0]
            y = current_srs[i][1] - final_srs[i][1]
            result.append([x,y])
        return result
    # initiate mino's mutable information
    def init_mino(self):
        self.current_rot = 0
        self.current_pos = [self.spawn_pos[0], self.spawn_pos[1]]
    # Based on its current [x, y] and current_rot, calculate the space where mino's blocks occupie 
    def current_space(self):
        occupied_space = []
        mino_data = self.block_data[self.current_rot] # 4 x 2 list
        for each_block in mino_data:
            x_position = self.current_pos[0] + each_block[0]
            y_position = self.current_pos[1] + each_block[1]
            occupied_space.append([x_position, y_position])
        return occupied_space

I_mino = tetrimino(1, I_layout, i_srs, i_spawn)
J_mino = tetrimino(2, J_layout, jlstz_srs, jlostz_spawn)
L_mino = tetrimino(3, L_layout, jlstz_srs, jlostz_spawn)
O_mino = tetrimino(4, O_layout, o_srs, jlostz_spawn)
S_mino = tetrimino(5, S_layout, jlstz_srs, jlostz_spawn)
T_mino = tetrimino(6, T_layout, jlstz_srs, jlostz_spawn)
Z_mino = tetrimino(7, Z_layout, jlstz_srs, jlostz_spawn)

minos = {
    1: I_mino,
    2: J_mino,
    3: L_mino,
    4: O_mino,
    5: S_mino, 
    6: T_mino,
    7: Z_mino
}

class controller:
    I_mino = tetrimino(1, I_layout, i_srs, i_spawn)
    J_mino = tetrimino(2, J_layout, jlstz_srs, jlostz_spawn)
    L_mino = tetrimino(3, L_layout, jlstz_srs, jlostz_spawn)
    O_mino = tetrimino(4, O_layout, o_srs, jlostz_spawn)
    S_mino = tetrimino(5, S_layout, jlstz_srs, jlostz_spawn)
    T_mino = tetrimino(6, T_layout, jlstz_srs, jlostz_spawn)
    Z_mino = tetrimino(7, Z_layout, jlstz_srs, jlostz_spawn)

    minos = {
        1: I_mino,
        2: J_mino,
        3: L_mino,
        4: O_mino,
        5: S_mino, 
        6: T_mino,
        7: Z_mino
    }

    # last_deleted is whether any line was deleted at last round
    # btb_ready is wheter last deletion was either t-spin or tetris
    def __init__(self):
        self.field = [[0 for i in range(10)] for j in range(23)]
        self.dropping_mino = controller.minos[random.randint(1, 7)] 
        self.next_minos = [] # 1 ~ 7
        self.score = 0
        self.lines_deleted_all = 0
        self.total_used = 0
        self.hold_mino_id = None # 1 ~ 7
        self.last_move_is_rotate = False
        self.ren = 0
        self.last_deleted = False
        self.btb_ready = False
        self.hold_used = False
        self.gameover = False
        self.next_minos_init()
        
    def reset(self):
        self.__init__()

    
    def next_round(self):
        self.dropping_mino.init_mino()
        self.dropping_mino = controller.minos[self.next_minos[0]]
        self.next_minos_update()
        self.last_move_is_rotate = False
        self.hold_used = False
        self.total_used += 1
        self.check_gameover()

        if not self.gameover:
            for i in range(random.randint(1, 4)):
                self.rotate(1)

    
    def check_gameover(self):
        if sum(self.field[1]) > 0:
            self.gameover = True
    
    # move dropping mino an offset specified by x, right(x>0) or left(x<0)
    def move_x(self, x): 
        x_position = self.dropping_mino.current_pos[0] + x
        y_position = self.dropping_mino.current_pos[1]
        if self.check_collision(self.dropping_mino.current_rot, [x_position, y_position]) == False: # won't collide
            self.dropping_mino.current_pos[0] += x
            self.last_move_is_rotate = False

    # rotate dropping mino after calculating its final position 
    # calculation is done using Super Rotation System used in tetris world rule
    def rotate(self, angle_offset):
        srs_offset = self.dropping_mino.srs_xy(angle_offset)
        for xy_offset in srs_offset: # xy_offset is [x, y]
            x_position = self.dropping_mino.current_pos[0] + xy_offset[0]
            y_position = self.dropping_mino.current_pos[1] - xy_offset[1] # minus because y axis is downward
            rot = (self.dropping_mino.current_rot + angle_offset) % 4
            if self.check_collision(rot, [x_position, y_position]) == False:
                self.dropping_mino.current_rot = rot
                self.dropping_mino.current_pos = [x_position, y_position]
                self.last_move_is_rotate = True
                break

    # check if the dropping mino would collide to either of wall, floor, or stacks, when it is located xy with angle of rot
    # xy is absolute position, and each x and y value can have negative values when the dropping mino is near the left wall 
    # rot is absolute angle:
    #        {0: initial state, 1: 90 degree clockwise, 2: 180 clockwise(counterclockwise), 3: 90 degree counterclockwise}
    def check_collision(self, rot, xy) -> bool:
        # get a list of [x, y], representing the location where blocks occupy
        # Note here we could not use dropping_mino.current_space(), 
        # because it calculates the space occupied based on current location and rotation angle.
        occupied_space = []
        mino_data = self.dropping_mino.block_data[rot] # 4 x 2 list
        for each_block in mino_data:
            x_position = xy[0] + each_block[0]
            y_position = xy[1] + each_block[1]
            occupied_space.append([x_position, y_position])

        for each_block in occupied_space:
            # wall and floor check
            if each_block[0] < 0 or each_block[0] > 9 or each_block[1] > 22:
                return True
            # stack check
            if self.field[each_block[1]][each_block[0]] > 0:
                return True
        return False

    # returns False, 0 if it drops, retuns True, score if it lands
    def soft_drop(self):
        x_position = self.dropping_mino.current_pos[0]
        y_position = self.dropping_mino.current_pos[1] + 1
        if self.check_collision(self.dropping_mino.current_rot, [x_position, y_position]) == False: # won't collide
            self.dropping_mino.current_pos[1] += 1
            self.last_move_is_rotate = False
            return False, 0, [], False, [0, 0]
        else:
            score, x_list, perfect_landed, pos = self.land()
            self.next_round()
            return True, score, x_list, perfect_landed, pos

    def hard_drop(self):
        x_position = self.dropping_mino.current_pos[0]
        y_position = self.dropping_mino.current_pos[1] + 1
        while self.check_collision(self.dropping_mino.current_rot, [x_position, y_position]) == False:
            y_position += 1
        self.dropping_mino.current_pos[1] = y_position - 1

        score, x_list, perfect_landed, pos = self.land()
        self.next_round()
        return True, score, x_list, perfect_landed, pos

    # return basis + btb + ren + all_deletion info, column_index
    def land(self):
        # update field
        space = self.dropping_mino.current_space()
        mino_color = self.dropping_mino.mino_id
        for xy in space:
            self.field[xy[1]][xy[0]] = mino_color
        pos = [self.dropping_mino.current_pos[0], self.dropping_mino.current_pos[1]]

        #line deletion 
        y_list = []
        x_list = []
        for xy in space:
            x_list.append(xy[0])
            y_list.append(xy[1])
        y_list = list(dict.fromkeys(y_list)) #remove duplication
        x_list = list(dict.fromkeys(x_list))
        y_list = [y for y in y_list if np.prod(self.field[y]) != 0]

        perfect_landed = True
        touched_space = []
        for x in x_list:
            lowest = -1
            for xy in space:
                if xy[0] == x:
                    if lowest < xy[1]:
                        lowest = xy[1]
            touched_space.append([x, lowest + 1])
        for xy in touched_space:
            if xy[1] < 23:
                if self.field[xy[1]][xy[0]] == 0:
                    perfect_landed = False

                

        lines_deleted = len(y_list)
        self.lines_deleted_all += lines_deleted
        if lines_deleted == 0:
            self.last_deleted = False
            self.ren = 0
            return 0, x_list, perfect_landed, pos
            
        else: # deletion required
            if self.last_deleted:
                self.ren += 1
            self.last_deleted = True

            # determine basis
            #scoring
            basis = 0
            t_spin_check = self.t_spin_checker()
            if t_spin_check == 0:
                self.score_not_t_spin(lines_deleted)
                basis = lines_deleted ** 2
            elif t_spin_check == 1:
                self.score_t_spin_mini(lines_deleted)
                basis = 8
            elif t_spin_check == 2:
                self.score_t_spin(lines_deleted)
                basis = lines_deleted + 16
            
            btb = 0
            if t_spin_check > 1 or lines_deleted == 4: #tetris or t-spin (/mini)
                if self.btb_ready:
                    btb = 1
                self.btb_ready = True
            else:
                self.btb_ready = False

            self.delete_lines(y_list)
            all_deleted = 0
            # the bottom line filled with 0 means all deletion occured
            if self.field[22] == [0 for i in range(10)]:
                all_deleted = 4
            
            print(">> CLEAR LINE:{}  Score:{}  Lines:{}  Clear:{}".format(self.score_text, self.score, self.lines_deleted_all, lines_deleted))
            return basis + btb + self.ren + all_deleted, x_list, perfect_landed, pos


                
    # 0 --> no t spin 1--> t spin mini 2 --> t spin
    def t_spin_checker(self) -> int:
        if self.dropping_mino.mino_id != t_index:
            return 0
        elif self.last_move_is_rotate != True:
            return 0
        else:
            diago_squares = []
            x = self.dropping_mino.current_pos[0]
            y = self.dropping_mino.current_pos[1]
            for i in [0, 2]:
                for j in [0, 2]:
                    diago_squares.append([x + i, y + j])
            filled_counter = 0
            wall_floor_counter = 0
            for xy in diago_squares:
                # wall and floor check
                if xy[0] < 0 or xy[0] > 9 or xy[1] > 22:
                    wall_floor_counter += 1
                # stack check
                elif self.field[xy[1]][xy[0]] > 0:
                    filled_counter += 1
            if filled_counter > 2:
                return 2
            elif wall_floor_counter + filled_counter > 2:
                return 1
            else:
                return 0

    # deleting field lines specified by y_list
    def delete_lines(self, y_list):
        temp_field = []
        for i in range(len(self.field)): # 0 ~ 22
            if i not in y_list:
                temp_field.append(self.field[i])
        zeros_inserted = len(self.field) - len(temp_field) # number of zero rows inserted
        for i in range(zeros_inserted):
            temp_field.insert(0, [0 for i in range(10)])
        self.field = temp_field
    
    def score_not_t_spin(self, num_lines):
        if num_lines < 4:
            gain = num_lines * 200 - 100
            self.score += gain
            self.score_text = "+" + str(gain)
        else:
            gain = 800
            self.score += gain
            self.score_text = "Tetris! +800"

    def score_t_spin(self, num_lines):
        if num_lines == 1:
            self.score += 800
            self.score_text = "T-Spin Single"
        if num_lines == 2:
            self.score += 1200
            self.score_text = "T-Spin Double"
        if num_lines == 3:
            self.score += 1600
            self.score_text = "T-Spin Triple"

        
    def score_t_spin_mini(self, num_lines):
        if num_lines == 1:
            self.score += 200
            self.score_text = "T-Spin mini single"
        elif num_lines == 2:
            self.score += 400
            self.score_text = "T-Spin mini double"

    # 7 out of 10 minos in next_minos should be different types
    def next_minos_init(self):
        self.next_minos = random.sample([1,2,3,4,5,6,7,8,9,10], 10)
        for i in range(len(self.next_minos)):
            if self.next_minos[i] > 7:
                self.next_minos[i] = random.randint(1,7)
                
    # maintain the rule of 7 out of 10
    def next_minos_update(self):
        popped_mino = self.next_minos.pop(0)
        flag = 0
        for i in range(len(self.next_minos)):
            if self.next_minos[i] == popped_mino:
                self.next_minos.append(random.randint(1,7))
                flag = 1
                break
        if flag == 0:
            self.next_minos.append(popped_mino)
    
    # exchange or hold
    def hold(self):
        if self.hold_used:
            pass
        else:
            if self.hold_mino_id == None:
                self.hold_mino_id = self.dropping_mino.mino_id
                self.next_round()
            else:
                self.last_move_is_rotate = False
                self.dropping_mino.init_mino()
                temp_next_mino_id = self.hold_mino_id
                self.hold_mino_id = self.dropping_mino.mino_id
                self.dropping_mino = controller.minos[temp_next_mino_id]
            self.hold_used = True
    
    def get_state(self):
        field_copy = [[1.0 if block > 0 else 0.5 for block in line] for line in self.field]
        space = self.dropping_mino.current_space()
        for x, y in space:
            field_copy[y][x] = 2
        # insert next minos info
        for i in range(5):
            l = [0 for j in range(10)]
            l[self.next_minos[i] - 1] = 1
            field_copy.insert(0, l)
        
        # insert hold mino info
        l = [0 for i in range(10)]
        if self.hold_mino_id != None:
            l[self.hold_mino_id - 1] = 1
        field_copy.insert(0, l)

        return field_copy
    
    # return the height of the field (1 ~ 23)
    def highest(self):
        ret = 0
        for i, line in enumerate(self.field):
            if sum(line) > 0:
                ret = 23 - i
                break
        return ret
    
    def highest_at(self, idx):
        ret = 0
        for i, line in enumerate(self.field):
            if line[idx] > 0:
                ret = 23 - i
                break
        return ret
        
    # return 0 ~ 10 (10 is impossible)
    def hole_columns(self):
        hole_count = 0
        for i in range(10):
            highest = self.highest_at(i)
            for j in range(highest):
                if self.field[22-j][i] == 0:
                    hole_count += 1
                    break
        return hole_count
    
    def hole_columns_at(self, idx_list):
        hole_count = 0
        for i in idx_list:
            highest = self.highest_at(i)
            for j in range(highest):
                if self.field[22-j][i] == 0:
                    hole_count += 1
                    break
        return hole_count

    def add_fire(self, fire):
        empty_index = random.randint(0, 9)
        added_stacks = [[8 if i != empty_index else 0 for i in range(10)] for i in range(fire)]
        self.field.extend(added_stacks)
        self.field = self.field[fire:]
