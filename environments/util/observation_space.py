import gym


class MultiAgentObservationSpace(list):
    def __init__(self, agents_observation_space):
        for agent_observation in agents_observation_space:
            assert isinstance(agent_observation, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        return [agent_observation_space.sample() for agent_observation_space in self._agents_observation_space]
