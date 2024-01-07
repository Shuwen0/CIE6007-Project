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
from sklearn.preprocessing import MinMaxScaler


class REFIT_Dataset(Dataset):
    def __init__(self, data_file_path: str, target_channel: int, window_size=120, target_size=120, stride=120, crop=None, scale='standard', flag='train'):
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
        assert scale in [None, 'standard', 'minmax']


        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.window_size = window_size
        self.target_size = target_size
        self.stride = stride
        self.scale = scale
        if scale == 'standard':
            self.scaler = StandardScaler()
        elif scale == 'minmax':
            self.scaler = MinMaxScaler()

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

        if self.scale == 'standard':

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

        elif self.scale == 'minmax':

            # extract the training part of aggregate power for scaler fitting (scaler expects 2D input)
            train_data = self.data_df.iloc[self.border1s[0]:self.border2s[0], :] # [power, appliance1, ...]
            # Min-Max Scaler
            self.scaler.fit(train_data.values)

            # As MinMaxScaler doesn't provide mean and std, we'll store min and range for labels
            self.label_min = self.scaler.data_min_[self.appliance_index]
            self.label_range = self.scaler.data_range_[self.appliance_index]

            # transform the entire dataset
            data = self.scaler.transform(self.data_df.values)
            
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


# new version
class Dataset_REFIT(Dataset):
    def __init__(self, data_path, flag='train', size=None,
                 features='LD', 
                 target='OT', scale=True, timeenc=0, freq='t', 
                 percent=100, max_len=-1, target_channel=1, train_all=True):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 480
            self.label_len = 0
            self.pred_len = self.seq_len
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        # self.timeenc = timeenc
        # self.freq = freq
        # self.percent = percent
        self.target_channel = int(target_channel) + 1
        self.data_path = data_path
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len + 1

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        df_raw = pd.read_csv(self.data_path)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('Time')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        # df_raw = df_raw[['Time'] + cols]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        print(num_train,num_vali,num_test)
        border1s = [0, num_train, len(df_raw) - num_test]
        # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        # border2s = [len(df_raw), num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # if self.set_type == 0:
        #     border2 = (border2 - self.seq_len) + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            # print(cols_data)
        elif self.features == 'S':
            df_data = df_raw[[self.target_channel]]
        elif self.features == 'LD': # load disaggregation
            cols_data = df_raw.columns[[1, self.target_channel]]
            df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            # data = np.log(df_data.values + 1)  # Adds 1 to prevent log(0)
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            # data = data * 1000

        # df_stamp = df_raw[['Time']][border1:border2]
        # df_stamp['Time'] = pd.to_datetime(df_stamp.Time)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['Time'], 1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['Time'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)
        print(data.shape)
        # self.data_x = data[border1:border2, 0].reshape((len(data[border1:border2, 0]), 1))
        # self.data_y = data[border1:border2, self.target_channel].reshape((len(data[border1:border2, self.target_channel]), 1))
        self.data_x = data[border1:border2, 0].reshape((-1, 1))
        self.data_y = data[border1:border2, 1].reshape((-1, 1))
        # self.data_stamp = data_stamp
        print(self.data_x.shape)

    def __getitem__(self, index):
        # feat_id = index // self.tot_len
        # s_begin = index % self.tot_len
        # s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len
        # seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        # seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        s_begin = index * self.seq_len
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end] # [window_size, 1]
        seq_y = self.data_y[s_begin:s_end] # [window_size, 1]
        # print(seq_x.shape, seq_y.shape)
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[s_begin:s_end]
            # print(seq_x)
            # seq_x = seq_x.reshape(-1, self.seq_len)
            # seq_y = seq_y.reshape(-1, self.seq_len)
            # seq_x_mark = seq_x_mark.reshape(-1, self.seq_len)
            # seq_y_mark = seq_y_mark.reshape(-1, self.seq_len)       
        return seq_x, seq_y

    def __len__(self):
        return int((len(self.data_x))/self.seq_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    

# debug (1 in server, 2 in laptop) Currently using 1
# train_dataset = REFIT_Dataset(data_file_path='../REFIT/New_Data/CLEAN_House2.csv', 
#                           target_channel=1, 
#                           window_size=120, target_size=120, stride = 120, 
#                           crop=1200, scale=True, flag='train')
# train_dataset = Dataset_REFIT(root_path='../REFIT/New_Data/', flag='train', size=[120,0,120],
#                  features='LD', data_path='CLEAN_House2.csv',
#                  target='OT', scale=True, timeenc=0, freq='t', 
#                  percent=100, max_len=-1, target_channel=2, train_all=True)
# print(train_dataset[0])
# print(train_dataset[0][0].shape, train_dataset[0][1].shape)# [window_size,1], scalar
# print(len(train_dataset))

# test_dataset = Dataset_REFIT(data_path='../REFIT/New_Data/CLEAN_House2.csv', 
#                              flag='test', 
#                              size=[120,0,120], # window_size, NA, target_size
#                              features='LD', # load disaggregation
#                              target='OT', # NA
#                              scale=True, # minmax scaler
#                              timeenc=0, # NA
#                              freq='t', # NA
#                              percent=100, # NA, could be of use in transfer learning
#                              max_len=-1, # NA
#                              target_channel=1, # target index, same as before
#                              train_all=True) # NA
# print(min(test_dataset[0][0]))
# print(min(test_dataset[5000][0]))
# print(min(test_dataset[9554][0]))
# print(test_dataset[0][0].shape, test_dataset[0][1].shape)# [window_size,1], scalar
# print(len(test_dataset))




    







       