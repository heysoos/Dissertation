import torch
import torch.nn as nn
import torch.nn.functional as F


class Rule(nn.Module):
    def __init__(self, BETA=1, RADIUS=1):
        super().__init__()
        self.beta = BETA
        self.radius = RADIUS
        Rk = 2*RADIUS + 1

        self.temp_adapt = False
        self.alpha = 0.5  # update rate
        self.h = 1e-2  # decorrelating coeffecient (increase temp)
        self.eps = 1e-2  # 2.00e-2  # correlating coefficient (decrease temp)
        self.D = 1. #2. * self.eps  # diffusion coef

        self.m_pow = 2.
        self.temp_pow = 1
        self.temp_kernel_size = 1

        self.trace_memory = 1 - 1e-3

        nearest_neighbours = torch.zeros(1, 1, Rk, Rk).cuda()
        # nearest_neighbours = torch.randn(1, 1, Rk, Rk).cuda()

        nearest_neighbours[:, :, RADIUS, :] = 1.
        nearest_neighbours[:, :, :, RADIUS] = 1.
        nearest_neighbours[:, :, RADIUS, RADIUS] = 0

        self.nearest_neighbours = nn.Parameter(nearest_neighbours, requires_grad=False)

        # normalize the temperature decay rate according to the connection density
        # self.eps = self.eps / torch.sqrt(self.nearest_neighbours.sum())
        # self.h = self.h * torch.sqrt(self.nearest_neighbours.sum())

    def forward(self, x, temp_adapt=False):

        s = x[:, [0], ...]
        tr = x[:, [1], ...]
        b = x[:, [-1], ...]
        shape = s.shape
        Rk = self.radius

        s_pad = F.pad(s, (Rk, Rk, Rk, Rk), mode='circular')
        # b_pad = F.pad(b, (Rk, Rk, Rk, Rk), mode='circular')

        Js = F.conv2d(s_pad, self.nearest_neighbours, padding=0)
        delta_e = 2 * s * Js

        definite_flip = delta_e <= 0
        p = torch.exp(-delta_e * b)
        p = torch.where(definite_flip, torch.ones_like(s), p)

        rand = torch.rand_like(s)

        dropout_mask = (torch.rand_like(s[0, 0]) > 0.5).unsqueeze(0).unsqueeze(0)
        flip = -2. * torch.logical_and(rand < p, dropout_mask) + 1

        if temp_adapt and torch.rand(1) > 0.9:

            # radius of temperature diffusion kernel
            temp_rad = self.temp_kernel_size*Rk
            temp_kernel_size = 2*temp_rad + 1
            pads = tuple([Rk * self.temp_kernel_size for i in range(4)])

            # pad (beta -> temperature) field
            b_tpad = F.pad(b, pads, mode='circular')
            T_pad = 1. / b_tpad
            T = T_pad[..., temp_rad:-temp_rad, temp_rad:-temp_rad]

            # get local magnetization
            # s_unfold = F.unfold(s_pad, 2*Rk + 1) # local observations
            # sm = (s_unfold.mean(dim=1)).reshape(shape) # means


            # take the difference of the local temperature and the mean of its neighbours
            diff_T = (F.avg_pool2d(T_pad, temp_kernel_size, stride=1) - T)

            # newT = self.h * sm ** 2 - self.eps * T + self.D * diff_T

            # compute a change in temperature
            # growth = self.h * sm.abs() ** self.m_pow
            # decay = self.eps * T ** self.temp_pow
            # diffusion = self.D * diff_T


            ###########################################
            # trying a different method that's not a function of T
            s_j = F.unfold(F.pad(tr, (Rk, Rk, Rk, Rk), mode='circular'), 2 * Rk + 1).mean(dim=1).reshape(shape)


            growth = self.h * s_j.abs()  # decorrelates if neighbourhood is correlated
            decay = self.eps * (1 - tr.abs()) ** self.m_pow # correlates if itself is decorrelated
            diffusion = self.D * diff_T # spreads
            ###########################################

            deltaT = growth - decay + diffusion

            newT = T + deltaT
            newT = (1 - self.alpha) * T + self.alpha * newT

            b = 1. / newT

        tr = self.trace_memory * tr + (1 - self.trace_memory) * s
        return torch.cat([(s * flip), tr, b], axis=1)

class isingCA(nn.Module):
    def __init__(self, BETA=1, RADIUS=2):
        super().__init__()
        self.radius = RADIUS

        self.rule = Rule(BETA, RADIUS)

    def initGrid(self, shape):
        '''
        returns a random initial state of the Ising model
        state[0, 0] = spins
        state[0, 1]= traces
        state[0, 2] = temperature

        :param shape: (resx, resy)
        :return: torch.Tensor of size (1, 3, resx, resy)
        '''
        rand = (torch.rand(1, 3, shape[0], shape[1]) > torch.rand(1)) * 2. - 1.
        rand[:, 1, ...] = torch.zeros_like(rand[:, 1, ...])
        rand[:, -1, ...] = torch.ones_like(rand[:, -1, ...]) * self.rule.beta

        return rand.cuda()

    def forward(self, x, temp_adapt=False):
        return self.rule(x, temp_adapt)
