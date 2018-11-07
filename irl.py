import numpy as np
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *


class IRLNet(nn.Module):
  def __init__(self, lr):
    super(IRLNet, self).__init__()
    self.lr = lr
    #self.conv1 = nn.Conv2d(1, 5, 1)
    #self.conv2 = nn.Conv2d(5, 8, 1)
    #self.conv3 = nn.Conv2d(8, 1, 1)
    self.fc1 = nn.Linear(25, 400)
    self.fc2 = nn.Linear(400, 300)
    self.fc3 = nn.Linear(300, 1)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    x = F.elu(self.fc2(x))
    x = F.elu(self.fc3(x))
    #x = F.relu(self.conv1(x))
    #x = F.relu(self.conv2(x))
    #x = F.relu(self.conv3(x))
    return torch.tensor(x, dtype=torch.double)

# TODO: Here I would ASSUME that P_a could be 1 and the NUM_STATES
# would be len(episode) because each state is a unique state, right?
def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T) 
  using dynamic programming

  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

  
  returns:
    p       Nx1 vector - state visitation frequencies
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T]) 

  for traj in trajs:
    mu[traj[0].cur_state, 0] += 1
  mu[:,0] = mu[:,0]/len(trajs)

  for s in range(N_STATES):
    for t in range(T-1):
      if deterministic:
        mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
      else:
        mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
  p = np.sum(mu, 1)
  return p

# TODO: This is fucked up and we can't just index an array by a 96x96 matrix...
def demo_svf(trajs, n_states):
  """
  compute state visitation frequences from demonstrations
  
  input:
    trajs   list of list of Steps - collected from expert
  returns:
    p       Nx1 vector - state visitation frequences   
  """

  p = np.zeros(n_states)
  for traj in trajs:
    for step in traj:
      p[step.cur_state] += 1
  p = p/len(trajs)
  return p

# Todo: Also fucked up because I don't have P_a... Can we get this from Gym?
# Gym should have everything defined about an MDP
# Perhaps we just make P_a = 1 or P_a = 0.9 to create noise?
def deep_maxent_irl(feat_map, P_a, gamma, trajs, lr, n_iters):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                       landing at state s1 when taking action 
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """

  N_STATES, _, N_ACTIONS = np.shape(P_a)

  # init nn model
  net = IRLNet(lr)

  # find state visitation frequencies using demonstrations
  mu_D = demo_svf(trajs, N_STATES)

  mu_D = torch.tensor(mu_D, requires_grad=True)

  # Init optimizer
  optimizer = optim.SGD(net.parameters(), lr=net.lr, weight_decay=1e-5)

  # training 
  for iteration in range(n_iters):
    if iteration % (n_iters/10) == 0:
      print ('Training Step: {}'.format(iteration))

    # compute the reward matrix
    rewards = net(feat_map)

    rewards_np = rewards.detach().numpy()
    # compute policy 
    _, policy = value_iteration.value_iteration(P_a, rewards_np, gamma, error=0.01, deterministic=True)

    # compute expected svf
    mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
    mu_exp = torch.tensor(mu_exp, requires_grad=True)
    # compute gradients on rewards:
    grad = mu_D - mu_exp
    grad = grad * -1

    # Clear gradient buffer
    optimizer.zero_grad()

    # apply gradients to the neural network
    rewards.backward(grad)

    # Gradient clipping to prevent exploding gradients -> NaN in multiplication
    nn.utils.clip_grad_norm_(net.parameters(), 100.0)
    
    optimizer.step()

  # Final forward pass and return
  rewards = net(feat_map).detach().numpy()


  return normalize(rewards)
