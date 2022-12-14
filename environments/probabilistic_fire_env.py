from environments.abstract_fire_env import AbstractFireEnv
import numpy as np
import random
height = 100
width = 100
D = 2
K = 0.05

def getNeighbors(point):
    neighbors = []
    min_x = max(0, point[1]-D)
    max_x = min(99, point[1]+D)
    min_y = max(0, point[0]-D)
    max_y = min(99, point[0]+D)

    for y in range(min_y, max_y): 
      for x in range(min_x, max_x):
        neighbors.append((y, x))
    return neighbors

class ProbabilisticFireEnv(AbstractFireEnv):

  def next_observation(self):

    probability_map = np.zeros(shape=(height,width), dtype=float)
    for row in range(self.height):
      for col in range(self.width):
        if self.observation[row,col] == 1:
          if self.fuel[row, col] > 0:
            self.fuel[row, col] -= 1
          else:
            self.observation[row,col] = 0

        elif self.observation[row,col] == 0 and self.fuel[row, col] > 0:
          neighboring_cells = getNeighbors((row, col))
          pnm = 1
          for neighboring_cell in neighboring_cells:
            if self.observation[neighboring_cell] == 1:
              dnmkl = np.array([a-b for a, b in zip(neighboring_cell, (row,col))])
              norm = np.sum(dnmkl**2)
              pnmkl0 = K/norm
              pnmklw = K*(dnmkl @ self.wind)/norm 
              pnmkl  = max(0, min(1, (pnmkl0+pnmklw)))
              pnm *= (1-pnmkl)
          pmn = 1 - pnm
          probability_map[row, col] = pmn

    self.observation[probability_map > np.random.rand(height,width)]  = 1

    return self.observation

  def reset_observation(self):
    center = [49, 49]
    self.observation = np.zeros(shape=(self.height, self.width), dtype=int)
    self.observation[center[0]-2:center[0]+2, center[1]-2:center[1]+2] = 1
    self.fuel = np.random.randint(low=15, high=20, size=(self.height, self.width))
    self.wind = np.random.uniform(low=-0.25, high=0.25, size=2)
    return self.observation