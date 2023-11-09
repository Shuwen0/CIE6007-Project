'''
This implementation is adapted from https://github.com/Ming-er/NeuralNILM_Pytorch/blob/main/nilmtk/disaggregate/attention_cnn_pytorch.py
The paper is available at http://e-press.dwjs.com.cn/dwjs/weixin/2021-45-9-3700.html
'''


# Package import
from __future__ import print_function, division
from warnings import warn
import torch
from torchsummary import summary
import torch.nn as nn



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class attention_cnn_Pytorch(nn.Module):
    def __init__(self, window_size=599):
        # Refer to "ZHANG C, ZHONG M, WANG Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring[C].The 32nd AAAI Conference on Artificial Intelligence"
        super(attention_cnn_Pytorch, self).__init__()
        self.window_size = window_size

        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 5), 0),
            nn.Conv1d(1, 30, 10, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(30, 30, 8, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 3), 0),
            nn.Conv1d(30, 40, 6, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(40, 50, 5, stride=1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(50, 50, 5, stride=1),
            nn.ReLU(True)
        )

        self.ca = ChannelAttention(in_planes=50, ratio=4)
        self.sa = SpatialAttention(kernel_size=7)

        self.dense = nn.Sequential(
            nn.Linear(50 * window_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x): 
        '''
        NOTE: 
        This function takes input x: [batch_size, windwo_size, 1] -- aggregate power, 
        returns: y: [batch_size, 1] -- device state (ON/OFF)

        x undergoes 5 convolutional layers and 2 attention layers + 1 output linear layer:
        1. k=(10 x num_appliances), s=1, n=30, ReLU
        2. k=(8 x num_appliances), s=1, n=30, ReLU
        3. k=(6 x num_appliances), s=1, n=40, ReLU
        4. k=(5 x num_appliances), s=1, n=50, ReLU
        5. k=(5 x num_appliances), s=1, n=50, ReLU
        6. channel attention + spatial attention
        7. linear layers: output 1024, ReLU
        8. output linear layer: output 1, Linear
        '''
        x = x.permute(0, 2, 1) # [batch_size, num_channel, window_length]
        x = self.conv(x) 
        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.dense(x.view(-1, 50 * self.window_size))
        return x.view(-1, 1)
    
    def weights_initialization(self):

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                # Xavier_uniform will be applied to conv1d and dense layer, to be sonsistent with Keras and Tensorflow
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                     torch.nn.init.constant_(m.bias.data, val=0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



