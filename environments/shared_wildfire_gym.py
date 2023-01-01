from gym import Env
from gym import spaces
from environments.probabilistic_fire_env import ProbabilisticFireEnv
from environments.drone_env import DronesEnv
import numpy as np
from environments.util.action_space import MultiAgentActionSpace
from environments.util.observation_space import MultiAgentObservationSpace
HEIGHT = 100
WIDTH  = 100
DT     = 0.5      
DTI    = 0.1  

class SharedWildFireGym(Env):

    def __init__ (self, _n_agents = 2):
        self._n_agents = _n_agents

        self.action_space = MultiAgentActionSpace([spaces.Discrete(2) for _ in range(_n_agents)])

        self.observation_space =  MultiAgentObservationSpace([
            spaces.Tuple((
                spaces.Box(low=0, high=1.0, shape=(2, HEIGHT, WIDTH), dtype=np.float32),
                spaces.Box(low=-0.872665, high=0.872665, shape=(1,), dtype=np.float32),
                spaces.Box(low=0, high=141.421, shape=(1,), dtype=np.float32),
                spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                spaces.Box(low=-0.872665, high=0.872665, shape=(1,), dtype=np.float32)
            )),
            spaces.Tuple((
                spaces.Box(low=0, high=1.0, shape=(2, HEIGHT, WIDTH), dtype=np.float32),
                spaces.Box(low=-0.872665, high=0.872665, shape=(1,), dtype=np.float32),
                spaces.Box(low=0, high=141.421, shape=(1,), dtype=np.float32),
                spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
                spaces.Box(low=-0.872665, high=0.872665, shape=(1,), dtype=np.float32)
            )),

        ])
   
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
        return [drone.get_obs() for drone in self.dronesEnv.drones]


    def step (self, action_n):
        if self.done:
            # should never reach this point
            print("EPISODE DONE!!!")

        if self.time_steps % (DT//DTI) == 0:
            self.observation = self.fireEnv.step()
        
        for drone, action in zip(self.dronesEnv.drones, action_n):
            drone.step(action)

        rewards = self.dronesEnv.update(self.observation)
        self.done = not self.fireEnv.fire_in_range(6)
        
        observations = [drone.get_obs() for drone in self.dronesEnv.drones]

        return observations, rewards, [self.done]*2, self.info
   

    def render(self, fig, ax):
        self.dronesEnv.plot_drones(fig, ax[0])
        self.dronesEnv.plot_belief_map(fig, ax[1])
        self.dronesEnv.plot_time_elapsed(fig, ax[2])
        self.dronesEnv.plot_trajectory(fig, ax[3])
        self.fireEnv.plot_heat_map(fig, ax[4])
   


