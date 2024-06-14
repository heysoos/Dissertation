import numpy as np
import torch
import torch.nn as nn

class Rule(nn.Module):
    def __init__(self, CHANNELS=8, HIDDEN=16):
        super().__init__()
        self.channels = CHANNELS
        self.hidden = HIDDEN


        ###########################################
        # for forward_perception
        self.ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
        self.sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).cuda() / 8.0
        self.lap = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12, 2.0], [1.0, 2.0, 1.0]]).cuda() / 16.0

        self.filters = [nn.Parameter(torch.randn(3, 3).cuda())
                        for i in range(2)]

        self.w1 = torch.nn.Conv2d(CHANNELS * 4, HIDDEN, 1)
        self.w2 = torch.nn.Conv2d(HIDDEN, CHANNELS, 1, bias=False)
        self.w2.weight.data.zero_()
        ###########################################

class CA(nn.Module):
    def __init__(self, CHANNELS=8, HIDDEN=16):
        super().__init__()
        self.channels = CHANNELS
        self.hidden = HIDDEN

        self.rule = Rule(CHANNELS, HIDDEN)

    def initGrid(self, BS, RES):
        self.psi = torch.cuda.FloatTensor(2 * np.random.rand(BS, self.channels, RES, RES) - 1)

    def seed(self, RES, n):
        seed = torch.FloatTensor(np.zeros((n, self.channels, RES, RES)))
        seed[:, 3:, RES // 2, RES // 2] = 1
        return seed

    def perchannel_conv(self, x, filters):
        '''filters: [filter_n, h, w]'''
        b, ch, h, w = x.shape
        y = x.reshape(b * ch, 1, h, w)
        y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
        y = torch.nn.functional.conv2d(y, filters[:, None])
        return y.reshape(b, -1, h, w)
    def perception(self, x):
        # filters = torch.stack([self.rule.ident, self.rule.sobel_x, self.rule.sobel_x.T, self.rule.lap])
        filters = [self.rule.ident, self.rule.sobel_x, self.rule.sobel_x.T, self.rule.lap]
        return self.perchannel_conv(x, torch.stack(filters))

    def forward_perception(self, x, dt=1, update_rate=0.5):
        b, ch, h, w = x.shape
        y = self.perception(x)

        y = torch.relu(self.rule.w1(y))
        y = self.rule.w2(y)

        update_mask = (torch.rand(b, 1, h, w) + update_rate).floor().cuda()
        y = dt * y * update_mask
        res = x + y
        return res
