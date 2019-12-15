# Adopted from https://github.com/tensorflow/models/blob/master/research/delf/delf/python/delf_v1.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# The attention_type determines whether the attention based feature aggregation
# is performed on the L2-normalized feature map or on the default feature map
# where L2-normalization is not applied. Note that in both cases, attention
# functions are built on the un-normalized feature map. This is only relevant
# for the training stage.
# Currently supported options are as follows:
# * use_l2_normalized_feature:
#   The option use_l2_normalized_feature first applies L2-normalization on the
#   feature map and then applies attention based feature aggregation. This
#   option is used for the DELF+FT+Att model in the paper.
# * use_default_input_feature:
#   The option use_default_input_feature aggregates unnormalized feature map
#   directly.
_SUPPORTED_ATTENTION_TYPES = ['use_l2_normalized_feature', 'use_default_input_feature']

# Supported types of non-lineary for the attention score function.
_SUPPORTED_ATTENTION_NONLINEARITY = ['softplus']


class DELF(nn.Module):
    """Attention Block layer implementation"""

    def __init__(self, numc_featmap, remain, attention_type=_SUPPORTED_ATTENTION_TYPES[0],
                 attention_nonlinear=_SUPPORTED_ATTENTION_NONLINEARITY[0], kernel=1):
        """
        Args:
            attention_type
                Type of the attention structure.
        """
        super(DELF, self).__init__()
        self.attention_type = attention_type
        self.attention_nonlinear = attention_nonlinear
        self.conv1 = nn.Conv2d(in_channels=numc_featmap, out_channels=numc_featmap//2,
                               kernel_size=kernel, dilation=1) # 512
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=numc_featmap//2, out_channels=1, kernel_size=kernel)
        self.softplus = nn.Softplus()
        self.remain = remain

    def weight_init(self, m):
        # 使用isinstance来判断m属于什么类型
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            # m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def init(self):
        self.weight_init(self.conv1)
        self.weight_init(self.conv2)

    def forward(self, x):

        if self.attention_type not in _SUPPORTED_ATTENTION_TYPES:
            raise ValueError('Unknown attention_type.')
        if self.attention_type == 'use_l2_normalized_feature':
            attention_feature_map = F.normalize(x, p=2, dim=1)
        elif self.attention_type == 'use_default_input_feature':
            attention_feature_map = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.attention_nonlinear not in _SUPPORTED_ATTENTION_NONLINEARITY:
            raise ValueError('Unknown attention non-linearity.')
        if self.attention_nonlinear == 'softplus':
            x = self.softplus(x)

        x = torch.mul(attention_feature_map, x.expand(-1, attention_feature_map.shape[1], -1, -1))

        # attention_outputs = self.perform_attention(attention_feature_map, feature_map)
        # attention_feat, attention_prob, attention_score = attention_outputs

        # # send partial features to netVLAD
        # N, C, H, W = attention_prob.shape
        # attention_score_flatten = attention_prob.view(N, C, -1)
        # values, indices = torch.topk(attention_score_flatten, int(H*W*self.remain))
        #
        # Nf, Cf, Hf, Wf = feature_map.shape
        # feature_map_flatten = feature_map.view(Nf, Cf, -1)
        #
        # indices = indices.expand(-1, Cf, -1)
        # feature_map_filtered = feature_map_flatten.gather(2, indices).unsqueeze(-1)

        return x

