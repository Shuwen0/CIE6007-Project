import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import argparse
import os
import numpy as np

# files of my implementation
from REFIT_Dataset import REFIT_Dataset, Dataset_REFIT
from seq2point import Seq2point
from transformer import TransformerSeq2Seq, TransformerSeq2Point
from attention_cnn import attention_cnn_Pytorch
from dataset_infos import params_appliance
from model_infos import params_model
from seq2seqCNN import Seq2seqCNN
from utils.metrics import metric

from train_reg import params_dataset

'''
This file loads an arbitrary model and train
-- input x: [batch_size, window_size, 1]
-- output y: [batch_size, window_size, 1] (seq2seq) or [batch_size, 1] (seq2point)
'''

# =========================================== model parameters ========================================
# Hyperparameters (default)
model = 'seq2seqCNN' # ['s2p', 'TransformerSeq2Seq', 'TransformerSeq2Point', 'attention_cnn_Pytorch','seq2seqCNN']
batch_size = params_model[model]['batch_size'] # [1000, 128]
learning_rate = params_model[model]['lr'] # [1e-3, 1e-4]
num_epochs = params_model[model]['num_epochs'] # [10, 100]
printfreq = params_model[model]['printfreq'] # [100, 100]
window_size = params_model[model]['window_size'] # [599, 480] 
crop = params_model[model]['crop'] # [None, None]
header = params_model[model]['header'] # [0, 0]
optimizer_name = params_model[model]['optimizer'] # ['Adam', 'Adam']
criterion_name = params_model[model]['criterion'] # ["BCEWithLogitsLoss", 'BCEWithLogitsLoss']  

# Only for s2p
if model == 's2p' or model == 'attention_cnn_Pytorch' or model == 'seq2seqCNN':
    offset = window_size // 2
    n_dense = params_model[model]['n_dense']
    transfer_cnn = params_model[model]['transfer_cnn']

# Only for TransformerSeq2seq
if model == 'TransformerSeq2Seq':
    d_model = params_model[model]['d_model']
    n_head = params_model[model]['n_head']
    num_encoder_layers = params_model[model]['num_encoder_layers']
    offset = None # NA, won't be used for dataset creation

if model == 'TransformerSeq2Point':
    d_model = params_model[model]['d_model']
    n_head = params_model[model]['n_head']
    num_encoder_layers = params_model[model]['num_encoder_layers']
    offset = window_size // 2


def remove_space(string):
    return string.replace(" ","")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# ======================================================== general setting ==============================================
def get_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network\
                                     for energy disaggregation - \
                                     network input = mains window; \
                                     network target = the states of \
                                     the target appliance.')
    parser.add_argument('--task',
                        type=str,
                        default='regression',
                        help='Task to train: classification or regression')
    parser.add_argument('--appliance_name',
                        type=remove_space,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='../REFIT/New_Data/',
                        help='this is the directory of the training samples')
    parser.add_argument('--dataset',
                        type=str,
                        default='REFIT',
                        help='this is the directory of the training samples')
    parser.add_argument('--building',
                        type=int,
                        default=4,
                        help='this is the index of the building')
    parser.add_argument('--pretrainedmodel_dir',
                        type=str,
                        default=None,
                        help='this is the directory of the pre-trained models')
    parser.add_argument('--save_dir',
                        type=str,
                        default='models/',
                        help='this is the directory to save the trained models')
    parser.add_argument('--save_model',
                        type=int,
                        default=-1,
                        help='Save the learnt model:\
                        0 -- not to save the learnt model parameters;\
                        n (n>0) -- to save the model params every n steps;\
                        -1 -- only save the learnt model params\
                        at the end of training.')
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help='Number of GPUs to use:\
                            n -- number of GPUs the system should use;\
                            -1 -- do not use any GPU.')
    return parser.parse_args()
args = get_arguments()

# given hyperparameters
appliance_name = args.appliance_name
dataset_name = args.dataset
building = args.building
appliance_channel = params_dataset[dataset_name][building][appliance_name]
task = args.task
data_dir = os.path.join('..', dataset_name, 'New_Data') # NOW: '../REFIT/New_Data/'
gpu = args.gpu
num_appliances = 1
save_results = True



# Device configuration
if gpu == -1:
    device = 'cpu'
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on ", device, flush=True)


# Dataset and DataLoader 
# ======================================================== 'CLEAN_House' is based on REFIT dataset ===============
test_dataset = Dataset_REFIT(data_path=os.path.join(data_dir, 'CLEAN_House' + str(building) + '.csv'), 
                             flag='test', 
                             size=[120,0,120], # window_size, NA, target_size
                             features='LD', # load disaggregation
                             target='OT', # NA
                             scale=True, # minmax scaler
                             timeenc=0, # NA
                             freq='t', # NA
                             percent=100, # NA, could be of use in transfer learning
                             max_len=-1, # NA
                             target_channel=appliance_channel, # target index, same as before
                             train_all=True) # NA
print("The size of total test dataset is: ", len(test_dataset), flush=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model: input [batch_size, window_size, 1] -> output [batch_size, num_appliances]
if model == 's2p':
    NILMmodel = Seq2point(window_length=window_size, n_dense=n_dense, num_appliances=num_appliances, transfer_cnn=transfer_cnn, cnn_weights=None).to(device)
elif model == 'seq2seqCNN':
    NILMmodel = Seq2seqCNN(window_length=window_size, n_dense=n_dense, target_length=window_size, transfer_cnn=transfer_cnn, cnn_weights=None).to(device)
elif model == 'TransformerSeq2Seq':
    NILMmodel = TransformerSeq2Seq(window_size=window_size, d_model=d_model, nhead=n_head, num_encoder_layers=num_encoder_layers).to(device)
elif model == 'TransformerSeq2Point':
    NILMmodel = TransformerSeq2Point(window_size=window_size, d_model=d_model, nhead=n_head, num_encoder_layers=num_encoder_layers).to(device)
elif model == 'attention_cnn_Pytorch':
    NILMmodel = attention_cnn_Pytorch(window_size=window_size)

# where the model weights are saved.
save_path = os.path.join('models', dataset_name+'_B'+str(building)+'_'+appliance_name+'_'+model+'_standard.pth')

# load the model weights
NILMmodel.load_state_dict(torch.load(save_path))


NILMmodel = NILMmodel.to(device)

# Train the model
def test():
 
    # Switch model to evaluation mode
    NILMmodel.eval()

    # # Initialize variables to store test metrics
    # test_mae = 0.0
    # test_mape = 0.0
    # test_mse = 0.0
    # num_batches = 0

    # all_inputs = []
    all_labels = []
    all_predictions = []

    # Validation
    with torch.no_grad():

        for inputs, targets in test_loader:
            inputs = inputs.to(device).to(torch.float)
            targets = targets.to(device).squeeze(2)
            predictions = NILMmodel(inputs)


            # # Calculate metrics for the current batch
            # batch_mae = torch.mean(torch.abs(predictions - targets))
            # batch_mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))) * 100
            # batch_mse = torch.mean((predictions - targets) ** 2)

            # # Aggregate the metrics
            # test_mae += batch_mae
            # test_mape += batch_mape
            # test_mse += batch_mse
            # num_batches += 1


            # Convert to lists and append to all_labels and all_predictions
            # all_inputs.extend(inputs.squeeze(2).cpu().numpy().tolist())
            all_labels.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    # convert labels and predictions to numpy array
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # calculate the un-normalized version based on avergae and standard deviation

    # =================================== z-score ==============================
    training_label_mean = test_dataset.scaler.mean_[1]
    training_label_std = test_dataset.scaler.scale_[1]

    unnormalized_all_labels = all_labels * training_label_std + training_label_mean
    unnormalized_all_predictions = all_predictions * training_label_std + training_label_mean


    # =================================== minmax ==================================== 
    # training_label_min = test_dataset.scaler.min_[1]
    # training_label_scale = test_dataset.scaler.scale_[1] # scale = (max - min) / (X.max(axis=0) - X.min(axis=0)) < 0！！

    # unnormalized_all_labels = all_labels / training_label_scale + training_label_min
    # unnormalized_all_predictions = all_predictions / training_label_scale + training_label_min

    # store the labels and predictions (normalized + unnormalized)
    if save_results:

        label_file = os.path.join('results','npy_file', 'normalized', dataset_name+'_B'+str(building)+'_'+appliance_name+'_standard.npy')
        preds_file = os.path.join('results','npy_file', 'normalized', dataset_name+'_B'+str(building)+'_'+appliance_name+'_'+model+'_standard.npy')

        unnormalized_label_file = os.path.join('results','npy_file', 'unnormalized', dataset_name+'_B'+str(building)+'_'+appliance_name+'_standard.npy')
        unnormalized_preds_file = os.path.join('results','npy_file', 'unnormalized', dataset_name+'_B'+str(building)+'_'+appliance_name+'_'+model+'_standard.npy')
        
        np.save(label_file, np.array(all_labels))
        np.save(preds_file, np.array(all_predictions))

        np.save(unnormalized_label_file, np.array(unnormalized_all_labels))
        np.save(unnormalized_preds_file, np.array(unnormalized_all_predictions))



    # Calculate the average metrics over all batches
    # test_mae /= num_batches
    # test_mape /= num_batches
    # test_mse /= num_batches
    
    # calculate the metrics (normalized + unnormalized)
    mae, mse, rmse, mape, mspe, smape, nd = metric(all_predictions, all_labels)

    # Print or log the test metrics
    print("Below are the metrics for the normalized version: ===========================================")
    print(f'Test MSE: {mse}', flush=True)
    print(f'Test MAE: {mae}', flush=True)
    print(f'Test MAPE: {mape}', flush=True)
    print(f'Test RMSE: {rmse}', flush=True)
    print(f'Test MSPE: {mspe}', flush=True)
    print(f'Test SMAPE: {smape}', flush=True)
    print(f'Test ND: {nd}', flush=True)

    # now the un-normalized version
    mae, mse, rmse, mape, mspe, smape, nd = metric(unnormalized_all_predictions, unnormalized_all_labels)

    # Print or log the test metrics
    print("Below are the metrics for the un-normalized version: ===========================================")
    print(f'Test MSE: {mse}', flush=True)
    print(f'Test MAE: {mae}', flush=True)
    print(f'Test MAPE: {mape}', flush=True)
    print(f'Test RMSE: {rmse}', flush=True)
    print(f'Test MSPE: {mspe}', flush=True)
    print(f'Test SMAPE: {smape}', flush=True)
    print(f'Test ND: {nd}', flush=True)

        
            


if __name__ == '__main__':
    print("This is the result of test_reg.py!!", flush=True)
    test()
