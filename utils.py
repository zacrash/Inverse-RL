import numpy as np
import math
from collections import namedtuple

Step = namedtuple('Step','cur_state action next_state reward done')


def normalize(vals):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_val = np.min(vals)
  max_val = np.max(vals)
  return (vals - min_val) / (max_val - min_val)


def sigmoid(xs):
  """
  sigmoid function
  inputs:
    xs      1d array
  """
  return [1 / (1 + math.exp(-x)) for x in xs]

""" Farbod and Zach proprietary matrix averager """
def voxelize(features):
  feat_T = features.T
  f = np.empty_like(features)
  x,y,d = features.shape
  x_per_cell, y_per_cell = x/100, y/100
  x_rem, y_rem = x % 100, y % 100
  
  for i in range(10-1):
    for j in range(10-1):
      avg[i][j][0] = np.sum(features[i*10:i*10+10][j*10:j*10+10][0])/10
      avg[i][j][1] = np.sum(features[i*10:i*10+10][j*10:j*10+10][1])/10
      avg[i][j][2] = np.sum(features[i*10:i*10+10][j*10:j*10+10][2])/10
      
      
    
