from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.Llama2_7B import Llama2_7B
from models.DLinear import DLinear


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
import pandas as pd

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Llama2_7B')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--freq', type=int, default=0)
parser.add_argument('--target', type=str, default='NSW', help='target feature in S or MS task')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48, help='start token length')

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--Llama2_layers', type=int, default=3)
parser.add_argument('--is_Llama2_7B', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--test_results_path', type=str, default='./test_results/')
parser.add_argument('--is_training', type=int, default=0)
parser.add_argument('--is_gen', type=int, default=0)
parser.add_argument('--is_test', type=int, default=0)
parser.add_argument('--base_model', type=str, default='NSW')
parser.add_argument('--is_gen_test', type=int, default=0)
parser.add_argument('--is_base_test', type=int, default=0)
# parser.add_argument('--target', type=str, default='NSW')

args = parser.parse_args()

SEASONALITY_MAP = {
   "minutely": 1440,
   "5_minutes": 288,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []
torch.cuda.empty_cache()
if args.is_training:

    for ii in range(args.itr):

        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_Llamal{}_df{}_eb{}_itr{}'.format(args.model_id, args.seq_len, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.Llama2_layers, 
                                                                        args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.freq == 0:
            args.freq = 'h'

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')
        # print(train_data.shape)
        # print(train_data, train_loader)

        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))

        device = torch.device('cuda:0')

        time_now = time.time()
        train_steps = len(train_loader)

        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        else:
            print(torch.cuda.device_count())
            model = Llama2_7B(args, device, path)
            model.to(device)
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            # mse, mae = test(model, test_data, test_loader, args, device, ii)
            # Check if multiple GPUs are available and wrap the model with DataParallel

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)
        
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        for epoch in range(args.train_epochs):

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for data in train_loader:
                print('Shape of first batch of data in train_loader:', data[0].size())
                break  # Exit after the first batch to just see one example

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                outputs = model(batch_x, ii)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                # print(outputs, batch_y)
                loss = criterion(outputs, batch_y)
                # print(loss)
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()

            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
            # mse, mae = test(model, test_data, test_loader, args, device, ii)
            # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    mses = np.array(mses)
    maes = np.array(maes)
    print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

if args.is_gen:
    print('gen....')
    for ii in range(args.itr):
        
        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_Llamal{}_df{}_eb{}_itr{}'.format(args.base_model, args.seq_len, args.label_len, args.pred_len,
                                                                            args.d_model, args.n_heads, args.e_layers, args.Llama2_layers, 
                                                                            args.d_ff, args.embed, ii)
        setting1 = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_Llamal{}_df{}_eb{}_itr{}'.format(args.model_id, args.seq_len, args.label_len, args.pred_len,
                                                                            args.d_model, args.n_heads, args.e_layers, args.Llama2_layers, 
                                                                            args.d_ff, args.embed, ii)
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        path1 = os.path.join(args.checkpoints, setting1)
        if not os.path.exists(path1):
            os.makedirs(path1)

        if args.freq == 0:
            args.freq = 'h'

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')
        # print(train_data.shape)
        # print(train_data, train_loader)

        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))

        device = torch.device('cuda:0')

        time_now = time.time()
        train_steps = len(train_loader)

        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        else:
            print(torch.cuda.device_count())
            model = Llama2_7B(args, device, path)
            model.to(device)
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            # mse, mae = test(model, test_data, test_loader, args, device, ii)
            # Check if multiple GPUs are available and wrap the model with DataParallel

        params = model.parameters()
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)
        
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if args.loss_func == 'mse':
            criterion = nn.MSELoss()
        elif args.loss_func == 'smape':
            class SMAPE(nn.Module):
                def __init__(self):
                    super(SMAPE, self).__init__()
                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
            criterion = SMAPE()
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        for epoch in range(args.train_epochs):

            iter_count = 0
            train_loss = []
            epoch_time = time.time()
            for data in train_loader:
                print('Shape of first batch of data in train_loader:', data[0].size())
                break  # Exit after the first batch to just see one example

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)

                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                outputs = model(batch_x, ii)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                # print(outputs, batch_y)
                loss = criterion(outputs, batch_y)
                # print(loss)
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()

            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
            # mse, mae = test(model, test_data, test_loader, args, device, ii)
            # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)
            early_stopping(vali_loss, model, path1)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    mses = np.array(mses)
    maes = np.array(maes)
    print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))

if args.is_test:
    ii = 0
    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_Llamal{}_df{}_eb{}_itr{}'.format(args.model_id, args.seq_len, args.label_len, args.pred_len,
                                                                        args.d_model, args.n_heads, args.e_layers, args.Llama2_layers, 
                                                                        args.d_ff, args.embed, ii)

    if args.freq == 0:
        args.freq = 'h'
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))

    path = os.path.join(args.checkpoints, setting)
    best_model_path = path + '/' + 'checkpoint.pth'
    device = torch.device('cuda:0')
    print('loading model from {}'.format(best_model_path))
    model = Llama2_7B(args, device, path)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    print("------------------------------------")
    mse, mae, preds, trues = test(model, test_data, test_loader, args, device, ii)
    mses.append(mse)
    maes.append(mae)
    np.save(args.test_results_path + args.data +'/pred_'+args.target+'.npy', preds)
    np.save(args.test_results_path + args.data +'/trues_'+args.target+'.npy', trues)