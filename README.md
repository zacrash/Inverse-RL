# Inverse Reinforcement Learning Research

This is a research repository, so the code is quite unorganized. Code was adapted from https://github.com/yrlu/irl-imitation and https://github.com/MatthewJA/Inverse-Reinforcement-Learning and converted to PyTorch with a few additions and functions to begin experiemnting with Open-AI gym.

## Requirements
* python 3.7
* PyTorch 1.0.0
* Numpy
* cvxopt
* Matplotlib

## Files
* Network design is in `irl.py`
* Main function is in `deep_maxent_irl.py`

## To run gridworld demo with PyTorch
`python deep_maxent_irl.py`
Use `-h` flag to view gridworld and training parameters

## NOTE: CURRENTLY BROKEN WITH PYTHON 3
