"""
Register all the necessary information to create a PIP package
"""
import os.path
import sys

from setuptools import setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym_bz_games'))

setup(
    name="gym_bz_games",
    version="1.0.0",
    author="yuanquan",
    author_email="yuanquan@gmail.com",
    description="Gym env of baseline3 zoo of game",
    url="https://github.com/pangafu/baseline3-zoo-games/",
    packages=['gym_bz_games', 'gym_bz_games.envs', 'gym_bz_games.wrappers'],


    install_requires=['gym>=0.18.0', 'gym-super-mario-bros >= 7.3.2', 'gym-tetris >= 3.0.2'],
    python_requires='>=3.6',
)
