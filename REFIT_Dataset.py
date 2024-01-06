# from Logger import log
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


class REFIT_Dataset_Old(Dataset):
    def __init__(self, filename, offset=299, window_size=599, crop=None, header=0, mode='s2p', flag='train', scale=True, percent=100, target_channel=2, mean=[522,700], std=[814,1000], normalize='not fixed'):

        self.filename = filename
        self.offset = offset
        self.header = header
        self.crop = crop
        self.mode = mode
        self.window_size = window_size
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.percent = percent
        self.target_channel = target_channel
        self.normalize = normalize
        self.mean = mean
        self.std = std

        self.df = pd.read_csv(self.filename,
                            nrows=self.crop,
                            header=self.header
                            )
        cols_data = self.df.columns[1:] ####################################### remove timestamps
        self.df = self.df[cols_data]
        # print(cols_data)
        
        self.scaler = StandardScaler()
        self.total_size = self.check_length()

        # seq2seq or seq2point
        if self.mode == 's2p' or self.mode == 'TransformerSeq2Point' or self.mode == 'attention_cnn_Pytorch': # seq2point needs padding on both sides
            self.available_size = self.total_size # each y_t is separate and unique
        elif self.mode == 'TransformerSeq2Seq' or self.mode == 'seq2seqCNN': # seq2seq
            self.available_size = self.total_size - self.window_size + 1 # predict y_{t:t+offset} each time

        # divide train/test/val
        num_train = int(self.available_size * 0.7)
        num_test = int(self.available_size * 0.2)
        num_vali = self.available_size - num_train - num_test

        # border1s contains the index of the first sample (0 means the first index)
        # birder2s contains the index of the last sample (open bracket)
        self.border1s = [0, num_train, self.available_size - num_test]
        self.border2s = [num_train, num_train + num_vali, self.available_size]
        self.border1 = self.border1s[self.set_type]
        self.border2 = self.border2s[self.set_type]

        # debug
        # print("=============================")
        # print("total size is ", self.total_size)
        # print("available size is ", self.available_size)
        # print("Boundary index are: ", self.border1s, self.border2s)
        # print("The start and end of this dataset instance are: ", self.border1, self.border2)

        self.pre_process() # defines self.x, self.y 

        # debug
        # print("x and y have length ", len(self.data_x), len(self.data_y))

        if self.mode == 's2p' or self.mode == 'TransformerSeq2Point' or self.mode == 'attention_cnn_Pytorch':
            self.padding() # padd zeros on both sides of self.x
            print("Padding completed successfully!")

    

    def check_length(self):
        return len(self.df)# the total number of rows of data
    
    def padding(self):

        # Calculate padding width using ceiling division
        padding_width = self.offset

        # convert to numpy array
        array = np.array(self.data_x)

        # Pad the array with zeros (self.data_x is one-dimensional array)
        self.data_x = np.pad(array, (padding_width, padding_width), mode='constant', constant_values=0)
    
    def pre_process(self):
        # normalize the data based on training statistics
        if self.normalize == 'fixed':
            data = self.df.values
            self.data_x = data[self.border1:+ self.window_size - 1, 0] # aggregate power
            self.data_y = data[self.border1:self.border2, self.target_channel] # appliance power
            self.data_x = (self.data_x - self.mean[0]) / self.std[0]
            self.data_y = (self.data_y - self.mean[1]) / self.std[1]
            

        else:
            if self.scale:
                train_data = self.df[self.border1s[0]:self.border2s[0]]
                # data = np.log(df_data.values + 1)  # Adds 1 to prevent log(0)
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(self.df.values)
            else:
                data = self.df.values
                # data = data * 1000
        
        if self.mode == 's2p' or self.mode == 'TransformerSeq2Point' or self.mode == 'attention_cnn_Pytorch':
            self.data_x = data[self.border1:self.border2, 0] # aggregate power
            self.data_y = data[self.border1:self.border2, self.target_channel] # appliance power
        else:
            self.data_x = data[self.border1:self.border2 + self.window_size - 1, 0]
            self.data_y = data[self.border1:self.border2 + self.window_size -  1, self.target_channel]

        # debug
        # print('df shape: ', self.df.shape)
        # print('data shape: ', data.shape)
        # print("x and y have shape ", self.data_x.shape, self.data_y.shape)
    
    def __len__(self):
        return self.border2 - self.border1

    def __getitem__(self, idx): # This idx is in alignment with midpoint (target), that is, 0 means the first midpoint

        
        if self.mode == 's2p' or self.mode == 'TransformerSeq2Point' or self.mode == 'attention_cnn_Pytorch':
            start_idx = idx # the real start_index of the input, starting from 0
            end_idx = idx + 2 * self.offset + 1 # couldn't be reached
            midpoint_idx = idx
            
            input = self.data_x[start_idx:end_idx].reshape((-1, 1)) # size: [window_size,1]
            target = self.data_y[midpoint_idx] # size: 1 (scaler)
            
            input_tensor = torch.from_numpy(input).float()
            target_tensor = torch.tensor(target, dtype=torch.float32) # torch.size () --> targets: [batch_size,]

            return input_tensor, target_tensor
    
        elif self.mode == 'TransformerSeq2Seq' or self.mode == 'seq2seqCNN':
            start_idx = idx
            end_idx = idx + self.window_size

            input = self.data_x[start_idx:end_idx].reshape(-1,1) # size: [window_size, 1] 
            # target = self.np_array[start_idx:end_idx,1].reshape(-1,1) # size: [window_size, 1]
            target = self.data_y[start_idx:end_idx] # size: [window_size,] 

            input_tensor = torch.from_numpy(input).float()
            target_tensor = torch.from_numpy(target).float()

            return input_tensor, target_tensor


import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class REFIT_Dataset(Dataset):
    def __init__(self, data_file_path: str, target_channel: int, window_size=120, target_size=120, stride=120, crop=None, scale=True, flag='train'):
        '''Dataset instance that reads the corresponding file (data and label)
        :param target_channel: int, column idx of appliance
        :param window_size: int, the size of the sliding window
        :param target_size: int, the size of the prediction window size, usually equal to window_size
        :param stride: int, the stride of the sliding window
        :param crop: the number of rows to read, None means all
        :param scale: bool, whether to scale aggregate power (z-score)
        :param flag: str, could be 'train'/'test'/'val', ratio = 7:2:1
        '''

        # ensure legal input
        assert flag in ['train', 'test', 'val']


        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.window_size = window_size
        self.target_size = target_size
        self.stride = stride
        self.scale = scale
        self.scaler = StandardScaler()

        # Read the files (for debug: crop=100)
        self.data_df = pd.read_csv(data_file_path, nrows=crop) # interval: 6-8s


        # debug ============================
        # check for the column names
        self.data_df.drop(columns=self.data_df.columns[0], axis=1, inplace=True) # drop the timestamp column
        cols_names = self.data_df.columns
        print(cols_names)
        
        # Get the column index for the specified appliance 
        # (timestamp column will be removed, so aggregate power is the first column)
        self.appliance_index = target_channel
        self.power_index = 0

        # each sample [x_t, x_{t+W-1}], calculate the total number of samples
        self.num_samples = (len(self.data_df) - self.window_size) // self.stride + 1 

        # divide train/test/val
        num_train = int(self.num_samples * 0.7)
        num_test = int(self.num_samples * 0.2)
        num_vali = self.num_samples - num_train - num_test

        self.length = [num_train, num_vali, num_test]

        self.border1s = [0, num_train*self.stride, (self.num_samples-num_test)*self.stride]
        self.border2s = [(num_train-1)*self.stride + self.window_size, # the end index of trainining data
                         (num_train+num_vali-1)*self.stride + self.window_size, # the end index of validation data
                         (self.num_samples-1)*self.stride + self.window_size] # the end index of testining data
        self.border1 = self.border1s[self.set_type]
        self.border2 = self.border2s[self.set_type]

        # debug:
        # print("self.border1s: ", self.border1s)
        # print("self.border2s: ", self.border2s)

        self.normalize()


    def __len__(self):
        return self.length[self.set_type]

    def __getitem__(self, idx): # idx means the idx of sample! idx = 0 means the first sample


        if idx < 0:  # Convert negative indices to positive
            idx = len(self) + idx

        if idx >= len(self):  # Check for out-of-bounds access
            raise IndexError("Index out of bounds")

        start_idx = idx * self.stride
        end_idx = idx * self.stride + self.window_size

        power = self.data_x[start_idx:end_idx].reshape(-1,1) # shape: [window_size, 1] 
        state = self.data_y[start_idx:end_idx] # shape: [window_size,] 

        power_tensor = torch.from_numpy(power).float()
        state_tensor = torch.from_numpy(state).float()

        return power_tensor, state_tensor
         
    def normalize(self):
        '''Normalize the aggregate and define x (power), y (state)'''

        if self.scale:

                # extract the training part of aggregate power for scaler fitting (scaler expects 2D input)
                train_data = self.data_df.iloc[self.border1s[0]:self.border2s[0], :] # [power, appliance1, ...]
                self.scaler.fit(train_data.values)

                # Store the mean and std dev for labels
                self.label_mean = self.scaler.mean_[self.appliance_index]
                self.label_std = self.scaler.scale_[self.appliance_index]

                # extract the column of aggregate power to be transformed
                data = self.scaler.transform(self.data_df.values)
                
                # # debug
                # print("The first row of training data is: ", data[0])
        else:
            data = self.data_df.values

            # no alterations, merely a placeholder
            self.label_mean = 0
            self.label_std = 1


        self.data_x = data[self.border1:self.border2, self.power_index]
        self.data_y = data[self.border1:self.border2, self.appliance_index]

        # debug
        # print("The length of aggregate power data in total is: ", self.data_x.shape[0])
        # print("The length of available samples is: ", self.border2-self.border1)


    

# # debug (1 in server, 2 in laptop) Currently using 1
# train_dataset = REFIT_Dataset(data_file_path='../REFIT/New_Data/CLEAN_House2.csv', 
#                           target_channel=1, 
#                           window_size=120, target_size=120, stride = 120, 
#                           crop=1200, scale=True, flag='train')
# print(train_dataset[0])
# print(train_dataset[0][0].shape, train_dataset[0][1].shape)# [window_size,1], scalar
# print(len(train_dataset))




    







       