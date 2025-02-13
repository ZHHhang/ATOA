import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
# import math




# DMLP Model-Path Generator 
class MLP(nn.Module):
	def __init__(self, input_size,num_sector,dropout_p=0):
		super(MLP, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(p=dropout_p),
		nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(p=dropout_p),
		nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(p=dropout_p),
		nn.Linear(896, 768),nn.PReLU(),nn.Dropout(p=dropout_p),
		nn.Linear(768, 512),nn.PReLU(),nn.Dropout(p=dropout_p),
		nn.Linear(512, 384),nn.PReLU()) #need PReLU?
		self.fc1=nn.Sequential(
		nn.Linear(384, 1280),nn.PReLU(), nn.Dropout(p=dropout_p),
		nn.Linear(1280, 640),nn.PReLU(), nn.Dropout(p=dropout_p),
		nn.Linear(640, 320),nn.PReLU(), nn.Dropout(p=dropout_p),
		nn.Linear(320, 160),nn.PReLU(), nn.Dropout(p=dropout_p),
		nn.Linear(160, 32),nn.PReLU(),nn.Dropout(p=dropout_p),
		nn.Linear(32, 16),nn.PReLU(),
		nn.Linear(16,1))
		self.fc2=nn.Sequential(
		nn.Linear(384, 1280),nn.PReLU(), nn.Dropout(p=dropout_p),
		nn.Linear(1280, 640),nn.PReLU(), nn.Dropout(p=dropout_p),
		nn.Linear(640, 320),nn.PReLU(), nn.Dropout(p=dropout_p),
		nn.Linear(320, 160),nn.PReLU(), nn.Dropout(p=dropout_p),
		nn.Linear(160, num_sector))  #if change 72, accuracy in data_loader_crossentropy also need to be change

		self.softmax_layer = nn.Softmax(dim=-1)
		print("mlp_hyper_parameter:")
		print("dropout_p")
		print(dropout_p)
		
		
		
        
	def forward(self,x):
		out_temp = self.fc(x)
		out_norm = self.fc1(out_temp)
		# out_norm = torch.relu(self.fc1(out_temp))
		out_orient_raw = self.fc2(out_temp)
		out_orient_softm = self.softmax_layer(out_orient_raw)
	
		#out_label=torch.argmax(out_orient_soft,dim=1)
		
		return out_norm, out_orient_raw, out_orient_softm


	
# DMLP Model-Path Generator 
class MLP_original(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP_original, self).__init__()
		self.fc = nn.Sequential(
		nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(),
		nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(),
		nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(),
		nn.Linear(896, 768),nn.PReLU(),nn.Dropout(),
		nn.Linear(768, 512),nn.PReLU(),nn.Dropout(),
		nn.Linear(512, 384),nn.PReLU(),nn.Dropout(),
		nn.Linear(384, 256),nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 256),nn.PReLU(), nn.Dropout(),
		nn.Linear(256, 128),nn.PReLU(), nn.Dropout(),
		nn.Linear(128, 64),nn.PReLU(), nn.Dropout(),
		nn.Linear(64, 32),nn.PReLU(),
		nn.Linear(32, output_size))
		
        
	def forward(self, x):
		out = self.fc(x)
		return out


 
