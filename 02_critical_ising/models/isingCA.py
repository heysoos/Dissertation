import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, BETA=1, CHANNELS=8, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.beta = BETA
        self.radius = RADIUS
        Rk = 2*RADIUS + 1

        xm, ym = torch.meshgrid(torch.linspace(-1, 1, Rk), torch.linspace(-1, 1, Rk))
        rm = torch.sqrt(xm ** 2 + ym ** 2).cuda()

        # nearest_neighbours = torch.ones(1, 1, Rk, Rk).cuda()
        # nearest_neighbours[:, :, RADIUS, RADIUS] = 0
        # nearest_neighbours[:, :, RADIUS + 1, RADIUS + 1] = 0.
        # nearest_neighbours[:, :, RADIUS + 1, RADIUS - 1] = 0.
        # nearest_neighbours[:, :, RADIUS - 1, RADIUS + 1] = 0.
        # nearest_neighbours[:, :, RADIUS - 1, RADIUS - 1] = 0.

        nearest_neighbours = torch.zeros(1, 1, Rk, Rk).cuda()
        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, :, :, RADIUS] = 1.
        # nearest_neighbours[:, :, -1, -1] = -1.
        # nearest_neighbours[:, :, 0, 0] = -1.
        # nearest_neighbours[:, :, 0, -1] = -1.
        # nearest_neighbours[:, :, -1, 0] = -1.
        nearest_neighbours[:, :, RADIUS, RADIUS] = 0


        # nearest_neighbours = nearest_neighbours.repeat(1, CHANNELS, 1, 1)

        # nearest_neighbours /= nearest_neighbours.norm()
        # nearest_neighbours[0, 1, :, :] = -nearest_neighbours[0, 1, :, :]
        # nearest_neighbours[0, 2, :, :] = -nearest_neighbours[0, 2, :, :]

        # nearest_neighbours = torch.zeros(CHANNELS, Rk, Rk)
        # nearest_neighbours[0, ]

        # nearest_neighbours = torch.ones(1, CHANNELS, Rk, Rk).cuda()
        # nearest_neighbours = nearest_neighbours.unsqueeze(0)
        # nearest_neighbours /= nearest_neighbours.norm()
        # nearest_neighbours[:, :, RADIUS, RADIUS] = 0.
        # nearest_neighbours[:, :, RADIUS + 1, RADIUS] = 0.
        # nearest_neighbours[:, :, RADIUS - 1, RADIUS] = 0.
        # nearest_neighbours[:, :, RADIUS, RADIUS - 1] = 0.
        # nearest_neighbours[:, :, RADIUS, RADIUS + 1] = 0.


        self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)
        # self.bias = nn.Parameter()

    def forward(self, x):

        Rk = self.radius
        s = F.pad(x, (Rk, Rk, Rk, Rk), mode='circular')
        Js = F.conv2d(s, self.nearest_neighbours, padding=0)
        delta_e = 2 * x * Js

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * self.beta)
        p = torch.where(definite_flip, torch.ones_like(x), p)

        rand = torch.rand_like(x)

        dropout_mask = (torch.rand_like(x[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        # flip = -2 * (rand < p) + 1


        return (x * flip)

class isingCA(nn.Module):
    def __init__(self, CHANNELS=1, BETA=1, RADIUS=2):
        super().__init__()
        self.channels = CHANNELS
        self.radius = RADIUS

        self.rule = Rule(BETA, CHANNELS, RADIUS)

    def initGrid(self, shape):
        rand = (np.random.rand(1, self.channels, shape[0], shape[1]) > 0.5) * 2. - 1.
        return torch.cuda.FloatTensor(rand)

    def forward(self, x):
        return self.rule(x)

    def cleanup(self):
        del self.psi
