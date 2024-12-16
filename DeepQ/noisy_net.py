import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer('epsilon_weight', torch.zeros((out_features, in_features)))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()
    
    # initialize parameters
    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
    
    
    def forward(self, input):
        # sample random noise in both weight and bias buffers
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        # perform linear transformation
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)


class NoisyDQN(nn.Module):
    def __init__(self, inshape, outshape):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inshape[1], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out_shape = self._get_conv_out_shape(inshape)
        self.noisy_layers = [
            NoisyLinear(conv_out_shape, 512),
            NoisyLinear(512, outshape),
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
        )
    
    def _get_conv_out_shape(self, inshape):
        return self.conv(torch.zeros(inshape)).shape[1]
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
    
    # signal-to-noise ratio = RMS(miu) / RMS(sigma)
    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight**2).mean().sqrt() / (layer.sigma_weight**2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]
