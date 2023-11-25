import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import os
import math
# import h5py

# global variables
transfer_dense = False
window_length = 599
n_dense = 1
num_appliances = 1

# padd the array to ensure the spatial dimension is not changed based on stride = 1
def get_same_padding(kernel_size):
    # For odd kernel sizes, this formula ensures 'same' padding
    if kernel_size % 2 == 1:
        return (kernel_size - 1) // 2
    # For even kernel sizes, you might choose to pad one less on the left
    else:
        return (kernel_size // 2 - 1, kernel_size // 2)

# model
class Seq2point(nn.Module):
    def __init__(self, window_length=599, n_dense=1, num_appliances=1, transfer_cnn=False, cnn_weights=None):
        super(Seq2point, self).__init__()
        self.conv1 = nn.Conv1d(1, 30, 10, stride=1)  # Even-sized kernel
        self.conv2 = nn.Conv1d(30, 30, 8, stride=1)   # Even-sized kernel
        self.conv3 = nn.Conv1d(30, 40, 6, stride=1)   # Even-sized kernel
        self.conv4 = nn.Conv1d(40, 50, 5, stride=1, padding=get_same_padding(5))   # Odd-sized kernel
        self.conv5 = nn.Conv1d(50, 50, 5, stride=1, padding=get_same_padding(5))   # Odd-sized kernel
        self.flatten = nn.Flatten()
        self.dense_layers = nn.ModuleList([nn.Linear(50 * window_length, 1024) for _ in range(n_dense)])
        self.output = nn.Linear(1024, num_appliances)

        self.num_appliances = num_appliances
        self.transfer_cnn = transfer_cnn # if True, load the weights from 'dense_weights' file
        self.cnn_weights = cnn_weights # xxxx.pth
        self.window_length = window_length

        self.weights_initialization()
        # Optionally load pretrained weights
        if self.transfer_cnn:
            self.load_pretrained_weights(self.cnn_weights)

    def forward(self, x):
        '''
        NOTE: 
        This function takes input x: [batch_size, windwo_size] -- aggregate power, 
        returns: y: [batch_size, 1] -- device state (ON/OFF)

        x undergoes 5 convolutional layers and n_dense dense layers + 1 output linear layer:
        1. k=(10 x num_appliances), s=1, n=30, ReLU
        2. k=(8 x num_appliances), s=1, n=30, ReLU
        3. k=(6 x num_appliances), s=1, n=40, ReLU
        4. k=(5 x num_appliances), s=1, n=50, ReLU
        5. k=(5 x num_appliances), s=1, n=50, ReLU
        6. flatten
        7. dense layers: output 1024, ReLU
        ......

        output: linear layer: output 1, Linear 
        '''


        x = x.view(x.size(0), -1, self.window_length) # [batch_size, num_channel, window_length]

        # PyTorch doesn't support assymmetric padding, use F.pad to manually padd for even kernels
        x = F.pad(x, get_same_padding(10))
        x = F.relu(self.conv1(x))

        x = F.pad(x, get_same_padding(8))
        x = F.relu(self.conv2(x))

        x = F.pad(x,get_same_padding(6))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.flatten(x)
        for dense in self.dense_layers:
            x = F.relu(dense(x))
        x = self.output(x)
        
        return x
    
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

    def load_pretrained_weights(self):

        pretrained_dict = torch.load(self.cnn_weights)
        model_dict = self.state_dict()
        
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        
        # Load the new state dict
        self.load_state_dict(model_dict)


# debug
# model = Seq2point(window_length=599, n_dense=1, num_appliances=1, transfer_cnn=False, cnn_weights=None)
# summary(model, (599,1))

