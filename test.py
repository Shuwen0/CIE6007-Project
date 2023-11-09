import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import argparse
import os
import numpy as np

# files of my implementation
from REFIT_Dataset import REFIT_Dataset
from seq2point import Seq2point
from transformer import TransformerSeq2Seq, TransformerSeq2Point
from attention_cnn import attention_cnn_Pytorch
from dataset_infos import params_appliance
from model_infos import params_model
from Focal_Loss import binary_focal_loss_with_logits

'''
This file loads an arbitrary model and train
-- input x: [batch_size, window_size, 1]
-- output y: [batch_size, window_size, 1] (seq2seq) or [batch_size, 1] (seq2point)
'''

# =========================================== model parameters ========================================
# Hyperparameters (default)
model = 's2p' # ['s2p', 'TransformerSeq2seq', 'TransformerSeq2Point', 'attention_cnn_Pytorch]
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
if model == 's2p' or model == 'attention_cnn_Pytorch':
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


use_focal_loss = False
alpha = 2

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
                        default='classification',
                        help='Task to train: classification or regression')
    parser.add_argument('--appliance_name',
                        type=remove_space,
                        default='kettle',
                        help='the name of target appliance'),
    parser.add_argument('--dataset_name',
                        type=str,
                        default="REFIT",
                        help="this is the name of the dataset (REFIT, UK_DALE, REDD, etc.)")
    parser.add_argument('--data_dir',
                        type=str,
                        default='../kettle/classification/kettle_testing_.csv',
                        help='this is the directory of the testing samples')
    parser.add_argument('--pretrainedmodel_dir',
                        type=str,
                        default='models/',
                        help='this is the directory of the pre-trained models')
    parser.add_argument('--save_dir',
                        type=str,
                        default='models/',
                        help='this is the directory to save the testing results')
    parser.add_argument('--save_results',
                        type=int,
                        default=-1,
                        help='Save the testing results:\
                        0 -- not to save;\
                        n (n>0) -- to save')
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
task = args.task
data_dir = '../' + appliance_name + '/' + task + '/' + appliance_name + '_test_H2.csv' # 最后要改！
gpu = args.gpu
num_appliances = 1
save_dir = args.save_dir
save_results = args.save_results
pretrainedmodel_dir = args.pretrainedmodel_dir
model_file = args.dataset_name + '_' + appliance_name + '_' + model + '.pth'




# Device configuration
if gpu == -1:
    device = 'cpu'
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================================================== initiate dataset & model =============================
# Dataset and DataLoader
test_dataset = REFIT_Dataset(filename=data_dir, offset=offset, window_size=window_size, crop=crop, mode=model)
print("The size of total testing dataset is: ", len(test_dataset))
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model: input [batch_size, window_size, 1] -> output [batch_size, num_appliances]
if model == 's2p':
    NILMmodel = Seq2point(window_length=window_size, n_dense=n_dense, num_appliances=num_appliances, transfer_cnn=transfer_cnn, cnn_weights=None).to(device)
elif model == 'TransformerSeq2Seq':
    NILMmodel = TransformerSeq2Seq(window_size=window_size, d_model=d_model, nhead=n_head, num_encoder_layers=num_encoder_layers).to(device)
elif model == 'TransformerSeq2Point':
    NILMmodel = TransformerSeq2Point(window_size=window_size, d_model=d_model, nhead=n_head, num_encoder_layers=num_encoder_layers).to(device)
elif model == 'attention_cnn_Pytorch':
    NILMmodel = attention_cnn_Pytorch(window_size=window_size)

# Load the weights from a file (.pth)
model_file = os.path.join(pretrainedmodel_dir, model_file)
state_dict = torch.load(model_file)
# Load the weights into the model
NILMmodel.load_state_dict(state_dict)

# Loss
if criterion_name == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss() # expect a single scalar output
elif criterion_name == 'BCELoss':
    criterion = nn.BCELoss() # expect multi-channel, needs softmax


if use_focal_loss and model == 's2p':
    aux_criterion_scaler = binary_focal_loss_with_logits # expect a single scalar output


# Train the model
def test():
    NILMmodel.eval()
    total_loss = 0
    TP = 0 # true positive
    FP = 0 # false positive
    TN = 0 # true negative
    FN = 0 # false positive
    num_points = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):

            num_points += inputs.shape[0]

            # debugging
            print("Testing sample ", i)
            print(inputs.shape, targets.shape)

            # Move tensors to the configured device
            inputs = inputs.to(device)

            # [batch_size, 1] for seq2point
            # [batch_size, window_size] for seq2seq
            targets = targets.to(device) 

            # Forward pass
            outputs = NILMmodel(inputs)
            batch_loss = criterion(outputs.type(torch.DoubleTensor), targets.type(torch.DoubleTensor).to(device))
            total_loss += batch_loss

            if use_focal_loss:
                aux_loss = aux_criterion_scaler(outputs, targets) * alpha
                total_loss += aux_loss

            # Calculate TP, TN, FP, FN
            np_outputs, np_targets = outputs.detach().cpu().numpy(), targets.detach().cpu().numpy()
            TP += np.sum((np_outputs == 1) & (np_targets == 1))
            TN += np.sum((np_outputs == 0) & (np_targets == 0))
            FP += np.sum((np_outputs == 1) & (np_targets == 0))
            FN += np.sum((np_outputs == 0) & (np_targets == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')
    print(f'Accuracy: {accuracy}')



if __name__ == '__main__':
    test()
