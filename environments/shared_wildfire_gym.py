from gym import Env
from gym import spaces
from environments.probabilistic_fire_env import ProbabilisticFireEnv
from environments.drone_env import DronesEnv
import numpy as np
HEIGHT = 100
WIDTH  = 100
DT     = 0.5      
DTI    = 0.1  

class SharedWildFireGym(Env):

    def __init__ (self, _n_agents = 2):
        self._n_agents = _n_agents
        self.action_space = spaces.Discrete(4) 
        self.observation_space = spaces.Dict(
            belief_map = spaces.Box(low=0, high=1.0, shape=(2, HEIGHT, WIDTH), dtype=np.float32),
            bank_angle = spaces.Box(low=-0.872665, high=0.872665, shape=(1,), dtype=np.float32),
            rho = spaces.Box(low=0, high=141.421, shape=(1,), dtype=np.float32),
            theta = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            psi = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            other_bank_angle = spaces.Box(low=-0.872665, high=0.872665, shape=(1,), dtype=np.float32),
        )
        self.fireEnv = ProbabilisticFireEnv(HEIGHT, WIDTH)
        self.dronesEnv = DronesEnv(HEIGHT, WIDTH, DT, DTI) 
        self.info = {}
        self.reset()

    @property
    def n_agents(self):
        return self._n_agents

    def reset(self):
        seed = self.fireEnv.reset()
        self.dronesEnv.reset(seed, self.fireEnv.observation)
        self.time_steps = 0
        self.done = False
        return self.get_obs()


    def get_obs(self):
        return {
            'belief_map': self.dronesEnv.observation, 
            'bank_angle': self.dronesEnv.drones[0].bank_angle,
            'rho':  self.dronesEnv.drones[0].rho,
            'theta':  self.dronesEnv.drones[0].theta,
            'psi':  self.dronesEnv.drones[0].psi,
            'other_bank_angle':  self.dronesEnv.drones[0].bank_angle,
        }

    def step (self, action_n):
        if self.done:
            # should never reach this point
            print("EPISODE DONE!!!")

        action_vector = [0, 0]
        if action_n == 1:
            action_vector = [0, 1]
        elif action_n == 2:
            action_vector = [1, 0]
        elif action_n == 3:
            action_vector = [1, 1]

            
        

        if self.time_steps % (DT//DTI) == 0:
            self.observation = self.fireEnv.step()
        
        rewards = self.dronesEnv.step(action_vector, self.observation)
        
        self.done = not self.fireEnv.fire_in_range(6)
        self.time_steps += 1

        return self.get_obs(), rewards, self.done, self.info
   

    def render(self, fig, ax):
        self.dronesEnv.plot_drones(fig, ax[0])
        self.dronesEnv.plot_belief_map(fig, ax[1])
        self.dronesEnv.plot_time_elapsed(fig, ax[2])
        self.dronesEnv.plot_trajectory(fig, ax[3])
        self.fireEnv.plot_heat_map(fig, ax[4])
   


