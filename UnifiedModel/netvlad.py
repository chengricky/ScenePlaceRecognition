"""
NetVLAD Module of Place Recognition Branch
It is based on
 https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py and
 https://github.com/Nanne/pytorch-NetVlad/blob/master/netvlad.py
Ricky 2019.Dec.14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""
    def __init__(self, num_clusters=64, dim=128, normalize_input=True, loop=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.loop = loop
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        index = faiss.IndexFlatL2(self.dim)
        index.add(traindescs)
        del traindescs
        dist, _ = index.search(clsts, 2)
        del index
        dsSq = torch.from_numpy(np.square(dist))
        # alpha - Parameter of initialization. Larger value is harder assignment.
        alpha = (-torch.log(torch.tensor(0.01)) / torch.mean(dsSq[:, 1] - dsSq[:, 0])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        del clsts, dsSq

        self.conv.weight = nn.Parameter((2.0 * alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(- alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        if self.loop:
            vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
            del x
            # slower than non-looped, but lower memory usage
            for C in range(self.num_clusters):
                residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                        self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
                residual *= soft_assign[:, C:C+1, :].unsqueeze(2)
                vlad[:, C:C+1, :] = residual.sum(dim=-1)
            del x_flatten, soft_assign
        else:
            residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                       self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign.unsqueeze(2)
            vlad = residual.sum(dim=-1)

        # intra-normalization
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(N, -1)  # flatten
        # L2 normalize
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad
