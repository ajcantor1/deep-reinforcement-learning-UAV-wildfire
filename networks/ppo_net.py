import torch
import torch.nn as nn
from networks.basedqn import BaseDQN
from torch.distributions import Categorical
import torch.nn.functional as F
class PPONet(BaseDQN):

  def __init__(self, _device, _channels, _height, _width, _outputs):

    super().__init__(_device, _channels, _height, _width, _outputs)

    self.fc1  = nn.Sequential(
      nn.Linear(5, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU()
    )

    self.conv = nn.Sequential(
      nn.Conv2d(2, 64, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2),
      nn.Conv2d(64, 64, kernel_size=3),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(2, stride=2)
    )
  
    conv_out_size = self._get_conv_out()

    self.fc2 = nn.Sequential(
      nn.Linear(conv_out_size, 500),
      nn.ReLU(),
      nn.Linear(500, 100),
      nn.ReLU(),
    )


    self.fc3 = nn.Sequential(
      nn.Linear(200, 200),
      nn.ReLU(),
    )
    
    self.ltsm = nn.LSTM(200, 200, batch_first=True)

    self.actor = nn.Linear(200, _outputs)

    self.critic = nn.Linear(200, 1)

  def forward(self, belief_map, state_vector, hidden=None):

    fc1_out = self.fc1(state_vector)
    conv_out = torch.flatten(self.conv(belief_map),1)
    fc2_out = self.fc2(conv_out)
    fc3_out = self.fc3(torch.cat((fc1_out, fc2_out), dim=1))

    ltsm_out = None
    new_hidden = None

    if hidden is not None:
      ltsm_out, new_hidden = self.ltsm(fc3_out, hidden)
    else:
      ltsm_out, new_hidden = self.ltsm(fc3_out)

    return self.actor(ltsm_out), self.critic(ltsm_out), new_hidden


