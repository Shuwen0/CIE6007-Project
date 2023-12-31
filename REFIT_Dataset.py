# from Logger import log
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


class REFIT_Dataset(Dataset):
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

            

    

# debug (1 in server, 2 in laptop) Currently using 1
# train_dataset = REFIT_Dataset(filename='../../../../NILM Datasets/REFIT/Raw_Data/CLEAN_House2.csv', 
#                           offset=299, 
#                           window_size=599, 
#                           crop=10, 
#                           header=0, 
#                           mode='s2p', 
#                           flag='train', 
#                           scale=True, 
#                           percent=100, 
#                           target_channel=8,
#                           normalize='not fixed')
# print(train_dataset[0])
# print(train_dataset[0][0].shape, train_dataset[0][1].shape)# [window_size,1], scalar
# print(len(train_dataset))

# train_dataset = REFIT_Dataset(filename='../kettle/Classification/kettle_training_.csv', offset=299, window_size=599, crop=600, mode='s2s')
# print(train_dataset[0]) # [window_size,1], [window_size,1], [window_size,1]
# print(train_dataset[0][0].shape, train_dataset[0][1].shape)
# print(len(train_dataset)) 
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)




    







       