import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple


import img_utils
from mdp import gridworld
from mdp import value_iteration
from irl import *
from maxent_irl import *
from utils import *
from lp_irl import *
from record_gym import *
import torch
import gym


###########################
## Example Traj ##
# [Step(cur_state, action, next_state, reward, done),...]
###########################


#####################
## Parse Arguments ##
#####################
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=200, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
PARSER.add_argument('-env', '--environment', default='Pong-v0', help='OpenAI gym environment')

ARGS = PARSER.parse_args()

GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters


def main():
    # Init Gym
    env = gym.make(ARGS.environment)
    OBS_S = env.observation_space.shape
    
    trajs = play(env)

    # use identity matrix as feature
    feat_map_np = env.reset()
    #feat_map_np = voxelize(feat_map_np)
    feat_map = torch.tensor(feat_map_np, dtype=torch.float)
    P_a = np.ones((210, 160, 3))

    print 'Deep Max Ent IRL training ..'
    rewards = deep_maxent_irl(feat_map, P_a, GAMMA, trajs, LEARNING_RATE, N_ITERS)

    # plots
    plt.figure(figsize=(20,4))
    plt.subplot(1, 4, 1)
    img_utils.heatmap2d(np.reshape(rewards_gt, (H,W), order='F'), 'Rewards Map - Ground Truth', block=False)
    plt.subplot(1, 4, 2)
    img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Reward Map - Recovered', block=False)
    plt.subplot(1, 4, 4)


if __name__ == "__main__":
  main()
