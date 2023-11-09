# from Logger import log
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class REFIT_Dataset(Dataset):
    def __init__(self, filename, offset=299, window_size=599, crop=None, header=0, mode='s2p'):

        self.filename = filename
        self.offset = offset
        self.header = header
        self.crop = crop
        self.mode = mode
        self.window_size = window_size

        self.df = pd.read_csv(self.filename,
                            nrows=self.crop,
                            header=self.header
                            )
        self.total_size = self.check_length()

        # seq2seq or seq2point
        if self.mode == 's2p' or self.mode == 'TransformerSeq2Point' or self.mode == 'attention_cnn_Pytorch': # seq2point needs padding on both sides
            self.np_array = self.padding(np.array(self.df)) # both sides padded with offset zeros
            self.available_size = self.total_size # each y_t is separate and unique
        elif self.mode == 'TransformerSeq2Seq': # seq2seq
            self.np_array = np.array(self.df)
            self.available_size = self.total_size - self.window_size + 1 # predict y_{t:t+offset} each time

    

    def check_length(self):
        return len(self.df)# the total number of rows of data
    
    def padding(self, array):

        # Calculate padding width using ceiling division
        padding_width = self.offset

        # Pad the array with zeros
        padded_array = np.pad(array, ((padding_width, padding_width), (0, 0)), mode='constant', constant_values=0)

        return padded_array
    
    def __len__(self):
        return self.available_size

    def __getitem__(self, idx): # This idx is in alignment with midpoint (target), that is, 0 means the first midpoint
        
        if self.mode == 's2p' or self.mode == 'TransformerSeq2Point' or self.mode == 'attention_cnn_Pytorch':
            start_idx = idx # the real start_index of the input, starting from 0
            end_idx = idx + 2 * self.offset + 1 # couldn't be reached
            midpoint_idx = idx + self.offset
            
            input = self.np_array[start_idx:end_idx,0].reshape((-1, 1)) # size: [window_size,1]
            target = self.np_array[midpoint_idx,1] # size: 1 (scaler)
            
            input_tensor = torch.from_numpy(input).float()
            target_tensor = torch.tensor([target], dtype=torch.float32)

            return input_tensor, target_tensor
    
        elif self.mode == 'TransformerSeq2Seq':
            start_idx = idx
            end_idx = idx + self.window_size

            input = self.np_array[start_idx:end_idx,0].reshape(-1,1) # size: [window_size, 1] 
            # target = self.np_array[start_idx:end_idx,1].reshape(-1,1) # size: [window_size, 1]
            target = self.np_array[start_idx:end_idx,1] # size: [window_size,] 

            input_tensor = torch.from_numpy(input).float()
            target_tensor = torch.from_numpy(target).float()

            return input_tensor, target_tensor

            

    

# debug
# train_dataset = REFIT_Dataset(filename='../kettle/Classification/kettle_training_.csv', offset=299, window_size=599)
# print(train_dataset[0]) # [window_size,1], scalar
# print(train_dataset[0][0].shape, train_dataset[0][1].shape)
# print(len(train_dataset))

# train_dataset = REFIT_Dataset(filename='../kettle/Classification/kettle_training_.csv', offset=299, window_size=599, crop=600, mode='s2s')
# print(train_dataset[0]) # [window_size,1], [window_size,1], [window_size,1]
# print(train_dataset[0][0].shape, train_dataset[0][1].shape)
# print(len(train_dataset)) 
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)




    







       