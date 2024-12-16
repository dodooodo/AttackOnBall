import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, inshape, outshape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inshape[1], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_shape = self._get_conv_out_shape(inshape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_shape, 512),
            nn.ReLU(),
            nn.Linear(512, outshape),
        )
    
    def _get_conv_out_shape(self, inshape):
        return self.conv(torch.zeros(inshape)).shape[1]
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)