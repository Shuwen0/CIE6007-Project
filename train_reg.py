import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os



import argparse

# files of my implementation
from REFIT_Dataset import REFIT_Dataset, Dataset_REFIT
from seq2point import Seq2point
from seq2seqCNN import Seq2seqCNN
from transformer import TransformerSeq2Seq, TransformerSeq2Point
from attention_cnn import attention_cnn_Pytorch
from dataset_infos import params_appliance
from model_infos import params_model
from utils.early_stopping import EarlyStopping

'''
This file loads an arbitrary model and train
-- input x: [batch_size, window_size, 1]
-- output y: [batch_size, window_size, 1] (seq2seq) or [batch_size, 1] (seq2point)
NOTE: to change a model, change the model name in this file & output_dir in run_train.sh
'''

# Hyperparameters (default)
model = 'seq2seqCNN' # ['s2p', 'TransformerSeq2Seq', 'TransformerSeq2Point', 'attention_cnn_Pytorch', 'seq2seqCNN']
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
    


params_dataset = {
    'REFIT':{
        2:{'kettle':8, 'microwave':5, 'fridge':1, 'dishwasher':3, 'washingmachine':2},
        3:{'kettle':9, 'microwave':8, 'fridge':2, 'dishwasher':5, 'washingmachine':6},
        5:{'kettle':8, 'microwave':7, 'fridge':1, 'dishwasher':4, 'washingmachine':3}
    }
}

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
                        default=2,
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


# Device configuration
if gpu == -1:
    device = 'cpu'
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on ", device, flush=True)

# Dataset and DataLoader 
# ======================================================== 'CLEAN_House' is based on REFIT dataset ===============
# train_dataset = REFIT_Dataset(data_file_path=os.path.join(data_dir, 'CLEAN_House' + str(building) + '.csv'), 
#                         target_channel=appliance_channel,
#                         crop=None, 
#                         flag='train', 
#                         scale=True)
# val_dataset = REFIT_Dataset(data_file_path=os.path.join(data_dir, 'CLEAN_House' + str(building) + '.csv'), 
#                         target_channel=appliance_channel,
#                         crop=None, 
#                         flag='val', 
#                         scale=True)
train_dataset = Dataset_REFIT(data_path=os.path.join(data_dir, 'CLEAN_House' + str(building) + '.csv'), 
                             flag='train', 
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
val_dataset = Dataset_REFIT(data_path=os.path.join(data_dir, 'CLEAN_House' + str(building) + '.csv'), 
                             flag='val', 
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
print("The size of total training dataset is: ", len(train_dataset), flush=True)
print("The size of validation dataset is: ", len(val_dataset))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

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

NILMmodel = NILMmodel.to(device)

# Loss and optimizer
criterion = torch.nn.MSELoss()

optimizer = optim.Adam(NILMmodel.parameters(), lr=learning_rate)

# Save the model parameters.
save_path = os.path.join('models', dataset_name+'_B'+str(building)+'_'+appliance_name+'_'+model+'.pth')

# Initialize the early stopping object
early_stopping = EarlyStopping(patience=3, verbose=True, path=save_path)

# Train the model
memory_flag = 0
def train():
    NILMmodel.train()
    print("Number of batches:", len(train_loader))
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_idx = 0
        for i, (inputs, targets) in enumerate(train_loader):

            # debugging
            # print("Training sample ", i)
            # print(inputs.shape, targets.shape)

            # Move tensors to the configured device
            inputs = inputs.to(device).to(torch.float)

            # [batch_size, 1] for seq2point
            # [batch_size, window_size, 1] for seq2seq
            # targets = targets.to(device) 

            # Forward pass
            if model == 'TransformerSeq2Seq-DE': # teacher-forcing
                tgt = targets.to(device).to(torch.float).squeeze(2) # [batch_size, window_size]
                start_token = 0
                start_tokens = torch.full((targets.shape[0], 1), start_token, device=device, dtype=targets.dtype)
                shifted_targets = torch.cat((start_tokens, tgt[:, :-1]), dim=1)  # Shape: [batch_size, seq_len]

                # Forward pass with teacher forcing
                # 'shifted_targets' is used as the second argument to the 'forward' method
                outputs = NILMmodel(inputs, shifted_targets)

            else:
                outputs = NILMmodel(inputs)


            # debug
            # print("outputs shape: ", outputs.shape)
            # loss = criterion(outputs.type(torch.DoubleTensor).to(device), targets.type(torch.DoubleTensor).to(device))
            loss = criterion(outputs.type(torch.DoubleTensor).to(device), targets.type(torch.DoubleTensor).to(device).squeeze(2))

            epoch_loss += loss
            epoch_idx += 1

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % printfreq == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f} (Average: {epoch_loss.item()/epoch_idx})', flush=True)
                # torch.save(NILMmodel.state_dict(), save_path)

                # # Check GPU memory usage
                # memory_allocated = torch.cuda.memory_allocated(device)
                # print(f"Epoch {epoch+1}: GPU memory allocated: {memory_allocated / (1024 ** 2):.2f} MB", flush=True)

                # # Monitor CPU usage
                # cpu_percent = psutil.cpu_percent(interval=1)  # Monitor CPU usage over 1 second
                # print(f"Epoch {epoch + 1}: CPU Usage: {cpu_percent}%", flush=True)

                # # Monitor memory usage
                # memory_info = psutil.virtual_memory()
                # print(f"Epoch {epoch + 1}: Memory Usage: {memory_info.percent}%", flush=True)
            
        # Switch model to evaluation mode
        NILMmodel.eval()

        # Validation
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in val_loader:
                inputs = inputs.to(device).to(torch.float)
                targets = targets.to(device)
                outputs = NILMmodel(inputs)
                # loss = criterion(outputs.type(torch.DoubleTensor).to(device), targets.type(torch.DoubleTensor).to(device))
                loss = criterion(outputs.type(torch.DoubleTensor).to(device), targets.type(torch.DoubleTensor).to(device).squeeze(2))
                val_loss += loss.item()

            val_loss /= len(val_loader)

        print(f"Validation Loss after epoch {epoch + 1}: {val_loss}")

        # Call the early stopping
        early_stopping(val_loss, NILMmodel)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Switch back to training mode
        NILMmodel.train()


        
            


if __name__ == '__main__':
    print("This is the result of train.py!!", flush=True)
    train()
