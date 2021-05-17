
from gym.envs.registration import register
from gym_bz_games.envs import Mario, Mario11, Mario12, Mario13, Mario14
from gym_bz_games.envs import Tetris


register(id='BZ-Mario-v0', entry_point='gym_bz_games:Mario')
register(id='BZ-Mario-1-1-v0', entry_point='gym_bz_games:Mario11')
register(id='BZ-Mario-1-2-v0', entry_point='gym_bz_games:Mario12')
register(id='BZ-Mario-1-3-v0', entry_point='gym_bz_games:Mario13')
register(id='BZ-Mario-1-4-v0', entry_point='gym_bz_games:Mario14')
register(id='BZ-Tetris-v0', entry_point='gym_bz_games:Tetris')

