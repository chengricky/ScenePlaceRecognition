from math import floor, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPooling2d(nn.Module):
    r"""apply spatial pyramid pooling over a 4d input(a mini-batch of 2d inputs
    with additional channel dimension) as described in the paper
    'Spatial Pyramid Pooling in deep convolutional Networks for visual recognition'
    Args:
        num_level: the output size of the pooling layer, list: i.e. [2,4,6,8] (the original scale is added defaultly)
        pool_type: max_pool, avg_pool, Default:max_pool
        overlap: whether to overlap when pooling
    By the way, the target output size is num_grid:
        num_grid = 0
        for i in range num_level:
            num_grid += (i + 1) * (i + 1)
        num_grid = num_grid * channels # channels is the channel dimension of input data
    examples:
        # >>> input = torch.randn((1,3,32,32), dtype=torch.float32)
        # >>> net = torch.nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1),\
        #                               nn.ReLU(),\
        #                               SpatialPyramidPooling2d(num_level=2,pool_type='avg_pool'),\
        #                               nn.Linear(32 * (1*1 + 2*2), 10))
        # >>> output = net(input)
    """

    def __init__(self, num_level, overlap=True, pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type
        self.overlap = overlap

    def forward(self, x):
        N, C, H, W = x.size()
        res = x.view(N, C, -1)
        for level in self.num_level:
            if not self.overlap:
                kernel_size = (ceil(H / level), ceil(W / level))
                stride = kernel_size
                padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))
            else:
                kernel_size = (ceil(2*H / (level+1)), ceil(2*W / (level+1)))
                stride = (ceil(H / (level+1)), ceil(W / (level+1)))
                remainder = (kernel_size[0]+(level-1)*stride[0]-H, kernel_size[1]+(level-1)*stride[1]-W)
                padding = (floor(remainder[0]/2), floor(remainder[1]/2))

            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, C, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, C, -1)

            res = torch.cat((res, tensor), 1)
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_level = ' + str(self.num_level) \
               + ', pool_type = ' + str(self.pool_type) + ')'


class SPPNet(nn.Module):
    def __init__(self, num_level=None, pool_type='max_pool'):
        super(SPPNet, self).__init__()
        if num_level is None:
            num_level = [2, 4, 6, 8]
        self.num_level = num_level
        self.pool_type = pool_type
        # self.feature = nn.Sequential(nn.Conv2d(3, 64, 3),
        #                              nn.ReLU(),
        #                              nn.MaxPool2d(2),
        #                              nn.Conv2d(64, 64, 3),
        #                              nn.ReLU())
        # self.num_grid = self._cal_num_grids(num_level)
        self.spp_layer = SpatialPyramidPooling2d(num_level)
        # self.linear = nn.Sequential(nn.Linear(self.num_grid * 64, 512),
        #                             nn.Linear(512, 10))

    def _cal_num_grids(self, level):
        count = 0
        for i in range(level):
            count += (i + 1) * (i + 1)
        return count

    def forward(self, x):
        # x = self.feature(x)
        x = self.spp_layer(x)
        # print(x.size())
        # x = self.linear(x)
        return x
