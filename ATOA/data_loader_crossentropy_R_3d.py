import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
import obstacle_handling_3d as oh_3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy import ndimage

import seaborn as sns

# Environment Encoder
length_obc = np.array([5,5,5,10,10,10,10,5,10,5])
width_obc = np.array([5, 10,10,5,5,10,10,5,10,5])
high_obc = np.array([10, 5,10,5,10,5,10,5,10,5])

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(6000, 786),nn.PReLU(),nn.Linear(786, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 60))

	def forward(self, x):
		x = self.encoder(x)
		return x
	
class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.decoder = nn.Sequential(nn.Linear(60, 256),nn.PReLU(),nn.Linear(256, 512),nn.PReLU(),nn.Linear(512, 786),nn.PReLU(),nn.Linear(786, 6000))
	def forward(self, x):
		x = self.decoder(x)
		return x
	
# 内插函数
def interpolate_points(p1, p2,distance_per_point):
    distance = np.linalg.norm(p1 - p2)
    num_points = int(distance / distance_per_point)

    return np.linspace(p1, p2, num_points + 2)

def get_effective_length(arr):
    # 检查数组维度
    if arr.ndim == 1:
        # 如果是一维数组，返回1（因为我们假设一维数组总是表示单一数据点）
        return 1
    else:
        # 对于多维数组，返回第一维的长度
        return len(arr)

#N=number of environments; NP=Number of Paths  //for orientation version
#def load_dataset_crossentropy(N=100,NP=4000):
def load_dataset_crossentropy(scale_para,bias,num_sector_theta,num_sector_phi,N=10,NP=4000,s=0,sp=0):	

	## Calculate the longest set of boundary points
	max_boundary_length=0
	boundary_lengths=np.zeros((N),dtype=int)
	for i in range(0,N):

		file_path = f"../../DATA_SET/r3d/obc_mask_normal_narrow/enviro_{i+s}.dat" # 根据需要更改文件名和路径
		# file_path = f"../../DATA_SET/r3d/obc_mask_3D/enviro_{i}.dat" # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para),int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)

		surface = oh_3d.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh_3d.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh_3d.surface_to_real(surface_points,bias,scale_para) 
		boundary_lengths[i]=len(boundary_points)
		if len(boundary_points)>max_boundary_length:
			max_boundary_length=len(boundary_points)
		# print("boundary_points",boundary_points,boundary_points.shape)
		# assert 0
		

	boundary_points_set=np.full((N,max_boundary_length,3),100, dtype=np.float32)   ## Take a very far point [100,100,100] so that it does not affect subsequent operations
	enviro_mask_set=np.zeros((N,int(40*scale_para), int(40*scale_para), int(40*scale_para)))

	obs_rep=np.zeros((N,60),dtype=np.float32)
	for i in range(0,N):
		#load obstacle point cloud

		file_path = f"../../DATA_SET/r3d/obc_mask_normal_narrow/enviro_{i+s}.dat" # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para),int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)
		
		enviro_mask_set[i]=enviro
		surface = oh_3d.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh_3d.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh_3d.surface_to_real(surface_points,bias,scale_para)  #numpy.ndarray
	
		
		for k in range(0,len(boundary_points)):
			boundary_points_set[i][k]=boundary_points[k]


	
	
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/r3d/Path_normal_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3) #change by z
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
			

	paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/r3d/Path_normal_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3) #change by z
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
	

	#orient and norm of the path by z 
	accuracy_theta=num_sector_theta #The number of spatial sectors  if change the value of accuracy, output_size in model also need to be change
	accuracy_phi=num_sector_phi
	orient_theta=np.zeros((N,NP,max_length-1),dtype=np.float32)
	orient_phi=np.zeros((N,NP,max_length-1),dtype=np.float32)
	norm=np.zeros((N,NP,max_length-1),dtype=np.float32)
	

	orient_theta_classification=np.zeros((N,NP,max_length-1,accuracy_theta),dtype=np.float32)
	orient_phi_classification=np.zeros((N,NP,max_length-1,accuracy_phi),dtype=np.float32)
	orient_theta_index=np.zeros((N,NP,max_length-1),dtype=int)
	orient_phi_index=np.zeros((N,NP,max_length-1),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/r3d/Path_normal_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path_reshape=path.reshape(len(path)//3,3) #change by z
		
				for k in range(0,len(path_reshape)-1):
					#Calculate the vector difference between adjacent points
					dx = path_reshape[k+1][0] - path_reshape[k][0]
					dy = path_reshape[k+1][1] - path_reshape[k][1]
					dz = path_reshape[k+1][2] - path_reshape[k][2]
					# 计算向量的模长
					r = math.sqrt(dx**2 + dy**2 + dz**2)
					# 计算θ (注意保证r不为零)
					if r == 0:
						theta = 0
					else:
						theta = math.acos(dz / r)  # acos输出范围在[0, π]
					# 计算ϕ
					phi = math.atan2(dy, dx)  # atan2输出范围在[-π, π]

					orient_theta[i][j][k] = math.degrees(theta)
					orient_phi[i][j][k] = math.degrees(phi) 
					norm[i][j][k]=r

					index_theta=int(orient_theta[i][j][k]//(180/accuracy_theta))
					index_phi=int(orient_phi[i][j][k]//(360/accuracy_phi)+accuracy_phi/2)
					orient_theta_classification[i][j][k][index_theta]=1.0
					orient_phi_classification[i][j][k][index_phi]=1.0
					orient_theta_index[i][j][k]=index_theta
					orient_phi_index[i][j][k]=index_phi
				

			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")	
	
		
			
			
				
	env_indices = []
	dataset=[]
	new_dataset=[]
	orient_dataset=[]
	targets=[]
	targets_future_all=[]
	# new_targets=[]
	classification_orient_theta_targets=[]
	classification_orient_phi_targets=[]
	classification_norm_targets=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:				
				for m in range(0, path_lengths[i][j]-1):
					data=np.zeros(66,dtype=np.float32)
					for k in range(0,60):
						data[k]=obs_rep[i][k]
					data[60]=paths[i][j][m][0]
					data[61]=paths[i][j][m][1]
					data[62]=paths[i][j][m][2]
					data[63]=paths[i][j][path_lengths[i][j]-1][0]
					data[64]=paths[i][j][path_lengths[i][j]-1][1]
					data[65]=paths[i][j][path_lengths[i][j]-1][2]
					
						
					targets.append(paths[i][j][m+1])
					#new_targets.append(orient_norm[i][j][m])
					#classification_orient_targets.append(orient_classification[i][j][m])
					classification_orient_theta_targets.append(orient_theta_index[i][j][m])
					classification_orient_phi_targets.append(orient_phi_index[i][j][m])
					classification_norm_targets.append(norm[i][j][m])
					dataset.append(data)
					env_indices.append(i)

					all_interpolated_points = []
					if path_lengths[i][j]>2:
						p1 = paths[i][j][m+1] #next point
						for n in range(m+1, path_lengths[i][j]-1):
							p2 = paths[i][j][n+1]
							interpolated_segment = interpolate_points(p1, p2, 0.2) #not include p1 but include p2
							all_interpolated_points.extend(interpolated_segment)
							# all_interpolated_points.extend(p2)
							p1 = p2
						if (m+1)==path_lengths[i][j]-1:
							all_interpolated_points.extend(p1)
						numpy_array_of_points = np.array(all_interpolated_points)  
						targets_future_all.append(numpy_array_of_points)
						# targets_future_all.append(numpy_array_of_points)
					elif path_lengths[i][j]>1:
						p1 = paths[i][j][m+1]
						all_interpolated_points.extend(p1)
						numpy_array_of_points = np.array(all_interpolated_points)  
						targets_future_all.append(numpy_array_of_points)
	
	max_length_targets_future_all = max(get_effective_length(arr) for arr in targets_future_all)	
	padded_targets_future_all = []
	for arr in targets_future_all:
		if arr.ndim == 1:
			pad_length = max_length_targets_future_all - len(np.expand_dims(arr, axis=0))  #Length to be filled
			padding_value = np.expand_dims(arr, axis=0)[-1,:]

		else:
			pad_length = max_length_targets_future_all - len(arr)  #Length to be filled
			padding_value = arr[-1,:]

		if pad_length > 0:
			padding_array = np.repeat([padding_value], pad_length, axis=0)
			padded_arr = np.vstack((arr, padding_array))
		else:
			padded_arr = arr
		padded_targets_future_all.append(padded_arr)

	new_dataset=dataset[:]
	orient_dataset=	dataset[:]	

	print("max_length_targets_future_all",max_length_targets_future_all)
	print("max_boundary_length",max_boundary_length)
	# assert 0
	

	orient_data=list(zip(env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_theta_targets,classification_orient_phi_targets,classification_norm_targets))
	random.shuffle(orient_data)
	env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_theta_targets,classification_orient_phi_targets,classification_norm_targets=zip(*orient_data)

	
	return 	enviro_mask_set,boundary_points_set,boundary_lengths,np.asarray(env_indices),np.asarray(dataset),np.asarray(targets),np.asarray(padded_targets_future_all),np.asarray(orient_dataset),np.asarray(classification_orient_theta_targets),np.asarray(classification_orient_phi_targets),np.asarray(classification_norm_targets) 







#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000

def load_test_dataset_crossentropy(scale_para,bias,num_sector_theta,num_sector_phi,N=10, NP=200,s=100, sp=0):	
	

	## Calculate the longest set of boundary points
	max_boundary_length=0
	boundary_lengths=np.zeros((N),dtype=int)
	for i in range(0,N):

		file_path = f"../../DATA_SET/r3d/obc_mask_normal_narrow/enviro_{i+s}.dat" # 根据需要更改文件名和路径
		# file_path = f"../../DATA_SET/r3d/obc_mask_3D/enviro_{i}.dat" # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para),int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)

		surface = oh_3d.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh_3d.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh_3d.surface_to_real(surface_points,bias,scale_para) 
		boundary_lengths[i]=len(boundary_points)
		if len(boundary_points)>max_boundary_length:
			max_boundary_length=len(boundary_points)
		# print("boundary_points",boundary_points,boundary_points.shape)
		# assert 0
		
	boundary_points_set=np.full((N,max_boundary_length,3),100, dtype=np.float32)   ## Take a very far point [100,100,100] so that it does not affect subsequent operations
	enviro_mask_set=np.zeros((N,int(40*scale_para), int(40*scale_para), int(40*scale_para)))

	obs_rep=np.zeros((N,60),dtype=np.float32)
	for i in range(0,N):

		file_path = f"../../DATA_SET/r3d/obc_mask_normal_narrow/enviro_{i+s}.dat" # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para),int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)
		
		enviro_mask_set[i]=enviro
		surface = oh_3d.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh_3d.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh_3d.surface_to_real(surface_points,bias,scale_para)  #numpy.ndarray
	
		
		for k in range(0,len(boundary_points)):
			boundary_points_set[i][k]=boundary_points[k]

		

	
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/r3d/Path_normal_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3) #change by z
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
			

	paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/r3d/Path_normal_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3) #change by z
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
	

	#orient and norm of the path by z 
	accuracy_theta=num_sector_theta #The number of spatial sectors  if change the value of accuracy, output_size in model also need to be change
	accuracy_phi=num_sector_phi
	orient_theta=np.zeros((N,NP,max_length-1),dtype=np.float32)
	orient_phi=np.zeros((N,NP,max_length-1),dtype=np.float32)
	norm=np.zeros((N,NP,max_length-1),dtype=np.float32)
	abnormal=[]

	orient_theta_classification=np.zeros((N,NP,max_length-1,accuracy_theta),dtype=np.float32)
	orient_phi_classification=np.zeros((N,NP,max_length-1,accuracy_phi),dtype=np.float32)
	orient_theta_index=np.zeros((N,NP,max_length-1),dtype=int)
	orient_phi_index=np.zeros((N,NP,max_length-1),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/r3d/Path_normal_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path_reshape=path.reshape(len(path)//3,3) #change by z
		
				for k in range(0,len(path_reshape)-1):
					#Calculate the vector difference between adjacent points
					dx = path_reshape[k+1][0] - path_reshape[k][0]
					dy = path_reshape[k+1][1] - path_reshape[k][1]
					dz = path_reshape[k+1][2] - path_reshape[k][2]
					# 计算向量的模长
					r = math.sqrt(dx**2 + dy**2 + dz**2)
					# 计算θ (注意保证r不为零)
					if r == 0:
						theta = 0
					else:
						theta = math.acos(dz / r)  # acos输出范围在[0, π]
					# 计算ϕ
					phi = math.atan2(dy, dx)  # atan2输出范围在[-π, π]

					orient_theta[i][j][k] = math.degrees(theta)
					orient_phi[i][j][k] = math.degrees(phi) 
					norm[i][j][k]=r

					index_theta=int(orient_theta[i][j][k]//(180/accuracy_theta))
					index_phi=int(orient_phi[i][j][k]//(360/accuracy_phi)+accuracy_phi/2)
					orient_theta_classification[i][j][k][index_theta]=1.0
					orient_phi_classification[i][j][k][index_phi]=1.0
					orient_theta_index[i][j][k]=index_theta
					orient_phi_index[i][j][k]=index_phi
				

			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")	
	
		
			
			
				
	env_indices = []
	dataset=[]
	new_dataset=[]
	orient_dataset=[]
	targets=[]
	targets_future_all=[]
	# new_targets=[]
	classification_orient_theta_targets=[]
	classification_orient_phi_targets=[]
	classification_norm_targets=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:				
				for m in range(0, path_lengths[i][j]-1):
					data=np.zeros(66,dtype=np.float32)
					for k in range(0,60):
						data[k]=obs_rep[i][k]
					data[60]=paths[i][j][m][0]
					data[61]=paths[i][j][m][1]
					data[62]=paths[i][j][m][2]
					data[63]=paths[i][j][path_lengths[i][j]-1][0]
					data[64]=paths[i][j][path_lengths[i][j]-1][1]
					data[65]=paths[i][j][path_lengths[i][j]-1][2]
					
						
					targets.append(paths[i][j][m+1])
					#new_targets.append(orient_norm[i][j][m])
					#classification_orient_targets.append(orient_classification[i][j][m])
					classification_orient_theta_targets.append(orient_theta_index[i][j][m])
					classification_orient_phi_targets.append(orient_phi_index[i][j][m])
					classification_norm_targets.append(norm[i][j][m])
					dataset.append(data)
					env_indices.append(i)

					all_interpolated_points = []
					if path_lengths[i][j]>2:
						p1 = paths[i][j][m+1] #next point
						for n in range(m+1, path_lengths[i][j]-1):
							p2 = paths[i][j][n+1]
							interpolated_segment = interpolate_points(p1, p2, 0.2) #not include p1 but include p2
							all_interpolated_points.extend(interpolated_segment)
							# all_interpolated_points.extend(p2)
							p1 = p2
						if (m+1)==path_lengths[i][j]-1:
							all_interpolated_points.extend(p1)
						numpy_array_of_points = np.array(all_interpolated_points)  
						targets_future_all.append(numpy_array_of_points)
						# targets_future_all.append(numpy_array_of_points)
					elif path_lengths[i][j]>1:
						p1 = paths[i][j][m+1]
						all_interpolated_points.extend(p1)
						numpy_array_of_points = np.array(all_interpolated_points)  
						targets_future_all.append(numpy_array_of_points)
	
	max_length_targets_future_all = max(get_effective_length(arr) for arr in targets_future_all)	
	padded_targets_future_all = []
	for arr in targets_future_all:
		if arr.ndim == 1:
			pad_length = max_length_targets_future_all - len(np.expand_dims(arr, axis=0))  #Length to be filled
			padding_value = np.expand_dims(arr, axis=0)[-1,:]

		else:
			pad_length = max_length_targets_future_all - len(arr)  #Length to be filled
			padding_value = arr[-1,:]

		if pad_length > 0:
			padding_array = np.repeat([padding_value], pad_length, axis=0)
			padded_arr = np.vstack((arr, padding_array))
		else:
			padded_arr = arr
		padded_targets_future_all.append(padded_arr)

	new_dataset=dataset[:]
	orient_dataset=	dataset[:]	

	print("max_length_targets_future_all",max_length_targets_future_all)
	print("max_boundary_length",max_boundary_length)
	# assert 0
	

	orient_data=list(zip(env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_theta_targets,classification_orient_phi_targets,classification_norm_targets))
	random.shuffle(orient_data)
	env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_theta_targets,classification_orient_phi_targets,classification_norm_targets=zip(*orient_data)

	return 	enviro_mask_set,boundary_points_set,boundary_lengths,np.asarray(env_indices),np.asarray(dataset),np.asarray(targets),np.asarray(padded_targets_future_all),np.asarray(orient_dataset),\
		np.asarray(classification_orient_theta_targets),np.asarray(classification_orient_phi_targets),np.asarray(classification_norm_targets) 
	 
	


def load_test_dataset_planner(scale_para,bias,N=10, NP=2000,s=100, sp=0): #change by z unseen environments
#def load_test_dataset_new(scale_para,bias,N=100,NP=200, s=0,sp=4000):'

	obs_start=s
	obc=np.zeros((N,10,3),dtype=np.float32)
	temp=np.fromfile('../../DATA_SET/r3d/obs.dat')
	obs=temp.reshape(len(temp)//3,3) #change by z

	temp=np.fromfile('../../DATA_SET/r3d/obs_perm2.dat',np.int32)
	perm=temp.reshape(184756,10)

	## loading obstacles
	for i in range(0,N):
		for j in range(0,10):
			for k in range(0,3):

				obc[i][j][k]=obs[perm[i+s][j]][k]
	
					
	# Q = Encoder()
	# D = Decoder()
	# Q.load_state_dict(torch.load('./AE_complex/models/cae_encoder_concave_2d_300_average loss_2.102830.pkl'))
	# D.load_state_dict(torch.load('./AE_complex/models/cae_decoder_concave_2d_300_average loss_2.102830.pkl'))
	# # Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	# # D.load_state_dict(torch.load('../models/cae_decoder.pkl'))
	# if torch.cuda.is_available():
	# 	Q.cuda()
	# 	D.cuda()
	

	## Calculate the longest set of boundary points
	max_boundary_length=0
	boundary_lengths=np.zeros((N),dtype=int)
	for i in range(0,N):
		temp=np.fromfile('../../DATA_SET/r3d/obs_cloud/obc'+str(i+s)+'.dat') 
		temp=temp.reshape(len(temp)//3,3)

		enviro = np.zeros((int(40*scale_para), int(40*scale_para),int(40*scale_para)))

		length_obc_mask=length_obc*scale_para
		width_obc_mask=width_obc*scale_para
		high_obc_mask=high_obc*scale_para
		obc_mask=(obc[i]+bias)*scale_para
		for j in range(0,10):
			center_x, center_y, center_z = obc_mask[j]
			# 计算障碍物的边界坐标
			left_x = int(np.floor(center_x - length_obc_mask[j] / 2))
			right_x = int(np.ceil(center_x + length_obc_mask[j] / 2))
			top_y = int(np.floor(center_y - width_obc_mask[j] / 2))
			bottom_y = int(np.ceil(center_y + width_obc_mask[j] / 2))
			lower_z = int(np.floor(center_z - high_obc_mask[j] / 2))
			upper_z = int(np.ceil(center_z + high_obc_mask[j] / 2))	

			for x in range(left_x, right_x + 1):
					for y in range(top_y, bottom_y + 1):
						for z in range(lower_z, upper_z + 1):	
							# if 0 <= x < enviro.shape[0] and 0 <= y < enviro.shape[1] and 0 <= z < enviro.shape[2]:
							enviro[x, y, z] = 1
		
		# obc_add = np.add(temp, bias)
		# obc_mul = obc_add * scale_para
		# for obc_p in obc_mul:
		# 	enviro[int(np.floor(obc_p[0])), int(np.floor(obc_p[1]))] = 1
		# 	enviro[int(np.floor(obc_p[0])), int(np.ceil(obc_p[1]))] = 1
		# 	enviro[int(np.ceil(obc_p[0])), int(np.floor(obc_p[1]))] = 1
		# 	enviro[int(np.ceil(obc_p[0])), int(np.ceil(obc_p[1]))] = 1
		
		enviro = ndimage.binary_fill_holes(enviro).astype(int) #mask
		surface = oh_3d.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh_3d.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh_3d.surface_to_real(surface_points,bias,scale_para) 
		boundary_lengths[i]=len(boundary_points)
		if len(boundary_points)>max_boundary_length:
			max_boundary_length=len(boundary_points)


	boundary_points_set=np.full((N,max_boundary_length,3),100, dtype=np.float32)   ## Take a very far point [100,100,100] so that it does not affect subsequent operations
	enviro_mask_set=np.zeros((N,int(40*scale_para), int(40*scale_para), int(40*scale_para)))


	# obs_rep=np.zeros((N,28),dtype=np.float32)	
	# obs_recover=np.zeros((N,8800),dtype=np.float32)

	for i in range(0,N): 
		temp=np.fromfile('../../DATA_SET/r3d/obs_cloud/obc'+str(i+s)+'.dat') 
		temp=temp.reshape(len(temp)//3,3)

		enviro = np.zeros((int(40*scale_para), int(40*scale_para),int(40*scale_para)))
		
		length_obc_mask=length_obc*scale_para
		width_obc_mask=width_obc*scale_para
		high_obc_mask=high_obc*scale_para
		obc_mask=(obc[i]+bias)*scale_para
		for j in range(0,10):
			center_x, center_y, center_z = obc_mask[j]
			# 计算障碍物的边界坐标
			left_x = int(np.floor(center_x - length_obc_mask[j] / 2))
			right_x = int(np.ceil(center_x + length_obc_mask[j] / 2))
			top_y = int(np.floor(center_y - width_obc_mask[j] / 2))
			bottom_y = int(np.ceil(center_y + width_obc_mask[j] / 2))
			lower_z = int(np.floor(center_z - high_obc_mask[j] / 2))
			upper_z = int(np.ceil(center_z + high_obc_mask[j] / 2))	

			for x in range(left_x, right_x + 1):
					for y in range(top_y, bottom_y + 1):
						for z in range(lower_z, upper_z + 1):	
							# if 0 <= x < enviro.shape[0] and 0 <= y < enviro.shape[1] and 0 <= z < enviro.shape[2]:
							enviro[x, y, z] = 1

		# obc_add = np.add(temp, bias)
		# obc_mul = obc_add * scale_para
		# for obc_p in obc_mul:
		# 	enviro[int(np.floor(obc_p[0])), int(np.floor(obc_p[1]))] = 1
		# 	enviro[int(np.floor(obc_p[0])), int(np.ceil(obc_p[1]))] = 1
		# 	enviro[int(np.ceil(obc_p[0])), int(np.floor(obc_p[1]))] = 1
		# 	enviro[int(np.ceil(obc_p[0])), int(np.ceil(obc_p[1]))] = 1

		enviro = ndimage.binary_fill_holes(enviro).astype(int) #mask
		enviro_mask_set[i]=enviro
		surface = oh_3d.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh_3d.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh_3d.surface_to_real(surface_points,bias,scale_para)  #numpy.ndarray
	
		for k in range(0,len(boundary_points)):
			boundary_points_set[i][k]=boundary_points[k]

		# obstacles=np.zeros((1,6000),dtype=np.float32)
		# obstacles[0]=temp.flatten()
		# inp=torch.from_numpy(obstacles)
		# inp=Variable(inp).cuda()
		# output=Q(inp)
		# output=output.data.cpu()
		# obs_rep[i]=output.numpy()z

		
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/r3d/Path/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3) #change by z
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")

		paths=np.zeros((N,NP,max_length,3), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/r3d/Path/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//3,3) #change by z
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
					


	return 	enviro_mask_set,obc,paths,path_lengths,obs_start
	

# obs_cloud_train just used for CAE trainning. do not use it in here
