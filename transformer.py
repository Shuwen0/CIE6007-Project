import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

from torch import nn, Tensor
import torch.optim as optim
import torch.utils.data as data
from typing import Optional, Any, Type, List, Tuple
from torch.nn.modules import (
    MultiheadAttention,
    Linear,
    Dropout,
    BatchNorm1d,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)

'''
This file implementes two methods: seq2seq and seq2point based on Transformer (encoder only)
Part of the framework is adapted from https://github.com/ludovicobuizza/HAR-Transformer/tree/main
'''

def get_activation_fn(activation_fn):
    if activation_fn == 'relu':
        return nn.ReLU()
    elif activation_fn == 'gelu':
        return nn.GELU()

def get_pos_encoder(pos_encoding: str) -> Type[nn.Module]:
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return PositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding)
    )




class PositionalEncoding(nn.Module):
    '''
    This module implements sin and cos positional encoding
    '''
    def __init__(self, window_size=128, d_model=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        P = torch.zeros(window_size, d_model) # [window_size, d_model]
        position = torch.arange(0, window_size, dtype=torch.float).unsqueeze(1) # [window_size, 1]: [[0,1,2,...]]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model/2]: [0, 2, 4, ...] * (-log(10000.0) / d_model)
        P[:, 0::2] = torch.sin(position * div_term) # [window_size, d_model/2]
        P[:, 1::2] = torch.cos(position * div_term) # [window_size, d_model/2]
        P = P.unsqueeze(0).transpose(0,1) # [window_size, 1, d_model]
        self.register_buffer('pe', P)

    def forward(self, X):
        '''
        NOTE: batch_size is the SECOND dimension of X!
        :param x: [window_size, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        '''

        X = X + self.pe[:X.size(0), :]
        return self.dropout(X)

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, window_size=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(window_size, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)







class TokenEmbedding(nn.Module):
    def __init__(self, window_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.linear = nn.Linear(1, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):

        # 对原始向量进行放缩出自论文3.4, prevent the positional encoding from 
        # being diminished when the embeddings have small values 
        # (which they often do when initialized)
        return self.linear(tokens) * math.sqrt(self.emb_size) 

class TransformerSeq2Point(nn.Module):
    def __init__(self, window_size=128, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        super(TransformerSeq2Point, self).__init__()

        '''
        :param window_size: the length of input sequence (aggregate power)
        :param d_model: d_k = d_v = d_model/n_head, default is 512
        :param nhad: numher of heads in multi-head attention, default is 8
        :param num_encoder_layers: number of encoder layers, default is 6
        :param num_decoder_layers: number of decoder layers, default is 6
        :param dim_feedforward: dimension of fully-connected layerm default is 2048
        :param dropout: dropout rate, default is 0.1
        '''

        # =================================== PE + TE (没检查)============================================
        self.d_model = d_model
        # Embedding layer that projects the input to d_model dimensions
        self.embedding = TokenEmbedding(window_size, d_model)
        # Positional Encoding layer
        self.pos_encoder = PositionalEncoding(window_size, d_model, dropout)

        # ==================================== Encoder ============================================
        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.act = get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.feat_dim = 1
        self.classifier = self.build_output_module(d_model=d_model, max_len=window_size, num_preds=1) 
    
    def build_output_module(
        self, d_model: int, max_len: int, num_preds: int
    ) -> nn.Module:
        """ Build linear layer that maps from d_model*max_len to num_classes.

        Softmax not included here as it is computed in the loss function.

        Args:
            d_model: the embed dim
            max_len: maximum length of the input sequence
            num_preds: the number of predictions (binary classification)

        Returns:
            output_layer: Tensor of shape (batch_size, num_classes)
        """
        output_layer = nn.Linear(d_model * max_len, num_preds)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, src):
        # src shape: [batch_size, window_size, 1]
        # Need to reshape and permute src to match the input requirement for the transformer which is (window_size, batch_size, feature_number)
        src = src.permute(1, 0, 2)
        # Pass the input through the embedding layer
        src = self.embedding(src)
        # Add the positional encoding
        src = self.pos_encoder(src)

        # Pass the input through the transformer encoder
        output = self.transformer_encoder(src)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, window_size, d_model)
        output = self.dropout1(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, window_size * d_model)
        output = self.classifier(output) # [batch_size, 1]  
        # probabilities = torch.sigmoid(output)

        # return probabilities
        return output
    
class TransformerSeq2Seq(nn.Module):
    def __init__(self, window_size=128, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        super(TransformerSeq2Seq, self).__init__()

        '''
        :param window_size: the length of input sequence (aggregate power)
        :param d_model: d_k = d_v = d_model/n_head, default is 512
        :param nhad: numher of heads in multi-head attention, default is 8
        :param num_encoder_layers: number of encoder layers, default is 6
        :param num_decoder_layers: number of decoder layers, default is 6
        :param dim_feedforward: dimension of fully-connected layerm default is 2048
        :param dropout: dropout rate, default is 0.1
        '''

        # =================================== PE + TE (没检查)============================================
        self.d_model = d_model
        # Embedding layer that projects the input to d_model dimensions
        self.embedding = TokenEmbedding(window_size, d_model)
        # Positional Encoding layer
        self.pos_encoder = PositionalEncoding(window_size, d_model, dropout)

        # ==================================== Encoder ============================================
        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.act = get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.feat_dim = 1
        self.classifier = self.build_output_module(d_model=d_model, max_len=window_size, num_preds=window_size) 
    
    def build_output_module(
        self, d_model: int, max_len: int, num_preds: int
    ) -> nn.Module:
        """ Build linear layer that maps from d_model*max_len to num_classes.

        Softmax not included here as it is computed in the loss function.

        Args:
            d_model: the embed dim
            max_len: maximum length of the input sequence
            num_preds: the number of predictions (binary classification)

        Returns:
            output_layer: Tensor of shape (batch_size, num_classes)
        """
        output_layer = nn.Linear(d_model * max_len, num_preds)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, src):
        # src shape: [batch_size, window_size, 1]
        # Need to reshape and permute src to match the input requirement for the transformer which is (window_size, batch_size, feature_number)
        src = src.permute(1, 0, 2)
        # Pass the input through the embedding layer
        src = self.embedding(src)
        # Add the positional encoding
        src = self.pos_encoder(src)

        # Pass the input through the transformer encoder
        output = self.transformer_encoder(src)
        output = self.act(
            output
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, window_size, d_model)
        output = self.dropout1(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, window_size * d_model)
        output = self.classifier(output) # [batch_size, window_size]  
        # probabilities = torch.sigmoid(output) # [batch_size, window_size]  

        # return probabilities
        return output
        
class TransformerSeq2Seq_DE(nn.Module):
    def __init__(self, window_size=128, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='gelu'):
        super(TransformerSeq2Seq_DE, self).__init__()

        '''
        :param window_size: the length of input sequence (aggregate power)
        :param d_model: d_k = d_v = d_model/n_head, default is 512
        :param nhad: numher of heads in multi-head attention, default is 8
        :param num_encoder_layers: number of encoder layers, default is 6
        :param num_decoder_layers: number of decoder layers, default is 6
        :param dim_feedforward: dimension of fully-connected layerm default is 2048
        :param dropout: dropout rate, default is 0.1
        '''

        # =================================== PE + TE (没检查)============================================
        self.d_model = d_model
        # Embedding layer that projects the input to d_model dimensions
        self.embedding = TokenEmbedding(window_size, d_model)
        # Positional Encoding layer
        self.pos_encoder = PositionalEncoding(window_size, d_model, dropout)

        # ==================================== Encoder ============================================
        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dropout1 = nn.Dropout(dropout)
        self.feat_dim = 1

        # ==================================== Decoder ============================================
        # Transformer Decoder
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)
        self.decoder_input = nn.Parameter(torch.zeros(window_size, 1, d_model))  # Initial input for the decoder

        # ===================================== output =========================================
        self.act = get_activation_fn(activation)
        self.classifier = self.build_output_module(d_model=d_model, max_len=window_size, num_preds=window_size) 
    
    def build_output_module(
        self, d_model: int, max_len: int, num_preds: int
    ) -> nn.Module:
        """ Build linear layer that maps from d_model*max_len to num_classes.

        Softmax not included here as it is computed in the loss function.

        Args:
            d_model: the embed dim
            max_len: maximum length of the input sequence
            num_preds: the number of predictions (binary classification)

        Returns:
            output_layer: Tensor of shape (batch_size, num_classes)
        """
        output_layer = nn.Linear(d_model * max_len, num_preds)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, src, tgt=None):
        # src shape: [batch_size, window_size, 1]
        # Need to reshape and permute src to match the input requirement for the transformer which is (window_size, batch_size, feature_number)
        src = src.permute(1, 0, 2)
        # Pass the input through the embedding layer
        src = self.embedding(src) # [window_size, batch_size, d_model]
        # Add the positional encoding
        src = self.pos_encoder(src)

        # Pass the input through the transformer encoder
        output = self.transformer_encoder(src)

        # If no target is provided, use repeated decoder input
        if tgt is None:
            tgt = self.decoder_input.expand(-1, src.size(1), -1)  # (window_size, batch_size, d_model)
        else:
            tgt = tgt.unsqueeze(2) # [batch_size, window_size, 1]
            tgt = tgt.permute(1,0,2) # [window_size, batch_size, 1]
            tgt = self.embedding(tgt)  # Embed the target sequence # [window_size, batch_size, d_model]
            tgt = self.pos_encoder(tgt)  # Add positional encoding

        memory = output  # Encoder output to be used as memory in the decoder
        output = self.transformer_decoder(tgt, memory)

        # Map the output of the decoder to the prediction space
        output = output.permute(1, 0, 2)  # (batch_size, window_size, d_model)
        output = output.reshape(output.shape[0], -1)  # (batch_size, window_size * d_model)
        output = self.classifier(output)  # [batch_size, window_size]

        # return probabilities
        return output

# debug
# Example usage:
# batch_size = 32
# window_size = 128
# src = torch.randn(batch_size, window_size, 1)
# model = TransformerSeq2Seq(window_size)
# logits = model(src)
# print(logits.shape)  # Expected shape: [batch_size, window_size, 1]