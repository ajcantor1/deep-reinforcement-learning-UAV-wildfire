from gym import Env
from gym import spaces
from environments.util.action_space import MultiAgentActionSpace
from environments.probabilistic_fire_env import ProbabilisticFireEnv
from environments.drone_env import DronesEnv

HEIGHT = 100
WIDTH  = 100
DT     = 0.5      
DTI    = 0.1  

class WildFireGym(Env):

    def __init__ (self, _n_agents = 2):
        self._n_agents = _n_agents
        self.action_space = [spaces.Discrete(2) for _ in range(self.n_agents)]
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
        states = []
        for i in range(self.n_agents):
            states.append({
                'state_vector': self.dronesEnv.drones[i].state,
                'belief_map': self.dronesEnv.drones[i].observation
            })      
        return states

    def step (self, action_n):
        if self.done:
            # should never reach this point
            print("EPISODE DONE!!!")

        assert len(action_n) == self.n_agents
        

        if self.time_steps % (DT//DTI) == 0:
            self.observation = self.fireEnv.step()
        
        rewards = self.dronesEnv.step(action_n, self.observation)
        
        self.done = not self.fireEnv.fire_in_range(6)
        self.time_steps += 1

        return self.get_obs(), rewards, self.done, self.info

    def render(self, fig, ax):
        self.dronesEnv.plot_drones(fig, ax[0])
        self.dronesEnv.plot_belief_map(fig, ax[1])
        self.dronesEnv.plot_time_elapsed(fig, ax[2])
        self.dronesEnv.plot_trajectory(fig, ax[3])
        self.fireEnv.plot_heat_map(fig, ax[4])
   


