import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

class Encoder_CNN_2D(nn.Module):
    def __init__(self,mask_size=160,dropout_p=0):
        super(Encoder_CNN_2D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p), 

            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_p),
        )

        self.fc_layers = nn.Sequential(
			nn.Linear(64 * 5 * 5, 512),
            # nn.Linear(512 * 9 * 9, 1024),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 28)
			# nn.Linear(1024, 28)
        )
        print("cnn_hyper_parameters:")
        print("mask_size")
        print(mask_size)
        print("dropout_p")
        print(dropout_p)
        # assert 0

    def forward(self, x):
    
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x



class Decoder_CNN_2D(nn.Module):
    def __init__(self, dropout_p=0):
        super(Decoder_CNN_2D, self).__init__()
        self.fc_layers = nn.Sequential(
            #Reverse fully connected layer
            nn.Linear(28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(512, 64 * 5 * 5),
            nn.BatchNorm1d(64 * 5 * 5),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.conv_layers = nn.Sequential(
            #Start upsampling by transposing convolutional layers
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),

        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), 64, 5, 5)  #Reorganize the shape matching convolution layer
        x = self.conv_layers(x)
        return x




class Encoder_CNN_2D_Layernorm(nn.Module):
    def __init__(self,mask_size=160,dropout_p=0):
        super(Encoder_CNN_2D_Layernorm, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([4, mask_size, mask_size]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([8, 80, 80]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([16, 40, 40]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 10, 10]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
			nn.Linear(64 * 5 * 5, 512),
            # nn.Linear(512 * 9 * 9, 1024),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 28)
			# nn.Linear(1024, 28)
        )
        print("cnn_hyper_parameters:")
        print("mask_size")
        print(mask_size)
        print("dropout_p")
        print(dropout_p)

    def forward(self, x):
    
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x