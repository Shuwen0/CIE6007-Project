import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        '''
        :param chomp_size: int, the number of elements needed removal
        '''
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        '''remove self.chomp_size elements from x
        x: [batch_size, 1, window_size]
        '''
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: list, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

class TCN(nn.Module):
    def __init__(self, window_size: int, channels: list, kernel_size=2, dropout=0.2, output='conv'):
        super(TCN, self).__init__()

        num_features = 1  # For NILM, the number of features is 1 (univariate time series)
        self.tcn = TemporalConvNet(num_features, channels, kernel_size=kernel_size, dropout=dropout)

        # Modify the final layer to output a sequence of the same length as the input window size
        if output == 'conv':
            self.output = nn.Conv1d(channels[-1], 1, kernel_size=1) # pointwise convolution
        elif output == 'linear':
            self.output = nn.Linear(channels[-1], window_size)

    def forward(self, input):
        # Reshape input to add a feature dimension ([batch_size, window_size] -> [batch_size, 1, window_size])
        input = input.unsqueeze(1)
        
        # Pass input through TCN
        features = self.tcn(input)

        # if TCN is only used to extract features, stop here and return the features
        if output == 'features':
            return features
        
        # Pass through final convolutional layer to get the desired output shape ([batch_size, 1, window_size])
        output = self.output(features).squeeze(1)  # Remove the singleton feature dimension

        return output

