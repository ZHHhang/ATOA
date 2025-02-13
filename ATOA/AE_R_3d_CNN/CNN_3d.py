import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

import torch.nn as nn

class Encoder_CNN_3D(nn.Module):
    def __init__(self, mask_size=160, dropout_p=0):
        super(Encoder_CNN_3D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(4), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p), 

            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),

            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
        )
        
         # Assuming mask_size reduces by half each time through the pooling layer
        reduced_size = mask_size // 2**5
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * reduced_size * reduced_size * reduced_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 60)
        )
        print("CNN_hyper_parameters:")
        print("mask_size:", mask_size)
        print("dropout_p:", dropout_p)
        # assert 0

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x







class Encoder_CNN_3D_duplicate(nn.Module):
    def __init__(self, mask_size=160, dropout_p=0):
        super(Encoder_CNN_3D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(4), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p), 

            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),

            # nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(16), 
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.Dropout(dropout_p),

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64), 
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
        )
        
         # Assuming mask_size reduces by half each time through the pooling layer
        reduced_size = mask_size // 2**5
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * reduced_size * reduced_size * reduced_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 60)
        )
        print("CNN_hyper_parameters:")
        print("mask_size:", mask_size)
        print("dropout_p:", dropout_p)
        # assert 0

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x










