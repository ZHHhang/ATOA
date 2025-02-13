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
import obstacle_handling as oh

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import ndimage

import seaborn as sns

# Environment Encoder
length_obc = np.array([2.0,7.5,5.0,5.0,6.0,4.3,4.5,5.0,3.8,4.5,4.9])
width_obc = np.array([7.0, 5.0,3.0,5.0,5.0,4.7,4.7,6.8,7.0,4.0,4.6])

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(8800,2200),nn.PReLU(),nn.Linear(2200, 512),nn.PReLU(),nn.Linear(512, 256),nn.PReLU(),nn.Linear(256, 128),nn.PReLU(),nn.Linear(128, 28))
			
	def forward(self, x):
		x = self.encoder(x)
		return x
	
class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.decoder = nn.Sequential(nn.Linear(28, 128),nn.PReLU(),nn.Linear(128, 256),nn.PReLU(),nn.Linear(256, 512),nn.PReLU(),nn.Linear(512, 2200),nn.PReLU(),nn.Linear(2200, 8800))
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
def load_dataset_crossentropy(scale_para,bias,num_sector,N=100,NP=4000,s=0, sp=0):

	# obc=np.zeros((N,11,2),dtype=np.float32)
	# temp=np.fromfile('../../DATA_SET/Concave_2D/obs.dat')
	# obs=temp.reshape(len(temp)//2,2) #change by z 
	# # np.set_printoptions(threshold=100)
	# temp=np.fromfile('../../DATA_SET/Concave_2D/obs_perm_concave.dat',np.int32)
	# perm=temp.reshape(167960,11)

	# # To visualize the demonstration path
	# for i in range(0,N):
	# 	os.makedirs('./Ground_Truth_Path_Img/e'+str(i), exist_ok=True)
	# 	for j in range(0,NP):
	# 		if os.path.isfile('../../DATA_SET/Concave_2D/obc_cloud_concave/obc_concave'+str(i)+'.dat'):
	# 			obs_plt=np.fromfile('../../DATA_SET/Concave_2D/obc_cloud_concave/obc_concave'+str(i)+'.dat')
	# 			obs_plt=obs_plt.reshape(len(obs_plt)//2,2)
	# 			path_plt=np.fromfile('../../DATA_SET/Concave_2D/Path/e'+str(i)+'/path'+str(j)+'.dat')	
	# 			path_plt=path_plt.reshape(len(path_plt)//2,2) #change by z		
	# 			plt.plot(path_plt[:, 0], path_plt[:, 1], marker='o',color='red', label='Path',markersize=3)
	# 			plt.scatter(obs_plt[:, 0], obs_plt[:, 1], color='blue',label='Obstacle Points',s=5)  
	# 			plt.title('Obstacle and Path Plot')
	# 			plt.xlabel('X')
	# 			plt.ylabel('Y')
	# 			plt.legend()
	# 			plt.show()
	# 			# plt.grid(True)
	# 			# plt.savefig('./Ground_Truth_Path_Img/e'+str(i)+'/enviroment_'+str(i)+'_and_path'+str(j)+'_plot.png')
	# 			plt.clf()
	# assert 0 
	#more function plz refer the plt_Ground_Truth_Path_Img_different_point.py

	
	# print("temp.shape")
	# print(temp.shape)

	## loading obstacles
	# for i in range(0,N):
	# 	for j in range(0,11):
	# 		for k in range(0,2):

	# 			obc[i][j][k]=obs[perm[i+s][j]][k]
	

	# Q = Encoder()
	# Q.load_state_dict(torch.load('./AE_complex/models/cae_encoder_concave_2d_300_average loss_2.102830.pkl'))
	# if torch.cuda.is_available():
	# 	Q.cuda()
	
	## Calculate the longest set of boundary points
	max_boundary_length=0
	boundary_lengths=np.zeros((N),dtype=int)
	for i in range(0,N):
		# temp=np.fromfile('../../DATA_SET/Concave_2D/obc_cloud_concave/obc_concave'+str(i+s)+'.dat') 
		# temp=temp.reshape(len(temp)//2,2)

		# enviro = np.zeros((int(40*scale_para), int(40*scale_para)))	
		# length_obc_mask=length_obc*scale_para
		# width_obc_mask=width_obc*scale_para
		# obc_mask=(obc[i]+bias)*scale_para
		# for j in range(0,11):
		# 	center_x, center_y = obc_mask[j]
		# 	#Calculate the coordinates of the four corners of the obstacle
		# 	left_x = int(np.floor(center_x - length_obc_mask[j] / 2))
		# 	right_x = int(np.ceil(center_x + length_obc_mask[j] / 2))
		# 	top_y = int(np.floor(center_y - width_obc_mask[j] / 2))
		# 	bottom_y = int(np.ceil(center_y + width_obc_mask[j]/ 2))
		# 	for x in range(left_x, right_x + 1):
		# 		for y in range(top_y, bottom_y + 1):
		# 			# if 0 <= x < enviro.shape[0] and 0 <= y < enviro.shape[1]:  # 确保标记不超出地图范围
		# 			enviro[x, y] = 1
		# #最外层是0，确保在障碍物代价loss的时候不会出错
		# enviro[0, :] = 0
		# enviro[-1, :] = 0
		# enviro[:, 0] = 0
		# enviro[:, -1] = 0
		# enviro = ndimage.binary_fill_holes(enviro).astype(int) #mask

		file_path = '../../DATA_SET/Concave_2D/obc_mask_concave_narrow/concave_enviro_' + str(i+s) + '.dat'  # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)
		# # 打印加载的数组和形状（可以根据需要移除）
		# plt.imshow(enviro.T, cmap='gray', origin='lower')
		# plt.show()
		# # assert 0
		# print("Loaded enviro for index", i, ":", enviro)
		# print("Shape:", enviro.shape)
		# print("enviro_loaded-enviro:", enviro-enviro)
		
		surface = oh.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh.surface_to_real(surface_points,bias,scale_para) 
		boundary_lengths[i]=len(boundary_points)
		if len(boundary_points)>max_boundary_length:
			max_boundary_length=len(boundary_points)

	boundary_points_set=np.full((N,max_boundary_length,2),100, dtype=np.float32)   ## Take a very far point [100,100] so that it does not affect subsequent operations
	enviro_mask_set=np.zeros((N,int(40*scale_para), int(40*scale_para)))
	# print(enviro_mask_set.dtype)


	obs_rep=np.zeros((N,28),dtype=np.float32)
	for i in range(0,N):
		#load obstacle point cloud
		# temp=np.fromfile('../../DATA_SET/Concave_2D/obc_cloud_concave/obc_concave'+str(i+s)+'.dat') 
		# temp=temp.reshape(len(temp)//2,2) #change by z

		# enviro = np.zeros((int(40*scale_para), int(40*scale_para)))
		# length_obc_mask=length_obc*scale_para
		# width_obc_mask=width_obc*scale_para
		# obc_mask=(obc[i]+bias)*scale_para
		# for j in range(0,11):
		# 	center_x, center_y = obc_mask[j]
		# 	#Calculate the coordinates of the four corners of the obstacle
		# 	left_x = int(np.floor(center_x - length_obc_mask[j] / 2))
		# 	right_x = int(np.ceil(center_x + length_obc_mask[j] / 2))
		# 	top_y = int(np.floor(center_y - width_obc_mask[j] / 2))
		# 	bottom_y = int(np.ceil(center_y + width_obc_mask[j]/ 2))
		# 	for x in range(left_x, right_x + 1):
		# 		for y in range(top_y, bottom_y + 1):
		# 			# if 0 <= x < enviro.shape[0] and 0 <= y < enviro.shape[1]:  # 确保标记不超出地图范围
		# 			enviro[x, y] = 1
		# #最外层是0，确保在障碍物代价loss的时候不会出错
		# enviro = ndimage.binary_fill_holes(enviro).astype(int) #mask
		# enviro[0, :] = 0
		# enviro[-1, :] = 0
		# enviro[:, 0] = 0
		# enviro[:, -1] = 0

		file_path = '../../DATA_SET/Concave_2D/obc_mask_concave_narrow/concave_enviro_' + str(i+s) + '.dat'  # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)
		# # 打印加载的数组和形状（可以根据需要移除）
		# print("Loaded enviro for index", i, ":", enviro)
		# print("Shape:", enviro.shape)
		# print("enviro_loaded-enviro:", enviro-enviro)

		enviro_mask_set[i]=enviro
		surface = oh.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh.surface_to_real(surface_points,bias,scale_para)  #numpy.ndarray
		# print("boundary_points")
		# print(boundary_points)
		# print(boundary_points.shape)
		# print(len(boundary_points))
		# print(type(boundary_points))
		# print("max_boundary_length")
		# print(max_boundary_length)
		# plt.scatter(boundary_points[:,0], boundary_points[:,1], c='red', s=1)
		# # # plt.scatter(temp[:,0], temp[:,1], c='green', s=1)
		# plt.savefig('./2D/narrow_point_outer_contour_img/obc'+str(i+s)+'.png')
		# plt.clf()
		# # assert 0
		
		for k in range(0,len(boundary_points)):
			boundary_points_set[i][k]=boundary_points[k]

		# obstacles=np.zeros((1,8800),dtype=np.float32)
		# obstacles[0]=temp.flatten()
		# inp=torch.from_numpy(obstacles)
		# inp=Variable(inp).cuda()
		# output=Q(inp)
		# output=output.data.cpu()
		# obs_rep[i]=output.numpy()
	# assert 0

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/Concave_2D/Path_concave_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2) #change by z
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)	
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
			

	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/Concave_2D/Path_concave_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2) #change by z
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
				
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
	

	#orient and norm of the path by z 
	accuracy=num_sector #The number of spatial sectors  if change the value of accuracy, outputsize in model also need to be change
	orient=np.zeros((N,NP,max_length-1),dtype=np.float32)
	norm=np.zeros((N,NP,max_length-1),dtype=np.float32)
	abnormal=[]
	# orient_norm=np.zeros((N,NP,max_length-1,2),dtype=np.float32)
	# orient_360=np.zeros((N,NP,max_length-1),dtype=np.float32)
	orient_classification=np.zeros((N,NP,max_length-1,accuracy),dtype=np.float32)
	orient_index=np.zeros((N,NP,max_length-1),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/Concave_2D/Path_concave_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path_reshape=path.reshape(len(path)//2,2) #change by z
				for k in range(0,len(path_reshape)-1):
					orient[i][j][k]=math.atan2(path_reshape[k+1][1]-path_reshape[k][1],path_reshape[k+1][0]-path_reshape[k][0])/(2*math.pi)*360
					norm[i][j][k]=math.sqrt(math.pow(path_reshape[k+1][1]-path_reshape[k][1],2)+math.pow(path_reshape[k+1][0]-path_reshape[k][0],2))
					# orient_norm[i][j][k][0]=orient[i][j][k]
					# orient_norm[i][j][k][1]=norm[i][j][k]
					# orient_360[i][j][k]=orient[i][j][k]+180.0
					
					index=int(orient[i][j][k]//(360/accuracy)+accuracy/2)
					if(index==accuracy):
						index=accuracy-1  #防止orient_classification恰好超出索引
					orient_classification[i][j][k][index]=1.0
					orient_index[i][j][k]=index
				
		

			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")	
	# print("111111")							
	env_indices = []
	dataset=[]
	new_dataset=[]
	orient_dataset=[]
	targets=[]
	targets_future_all=[]
	# new_targets=[]
	classification_orient_targets=[]
	classification_norm_targets=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:
							
				for m in range(0, path_lengths[i][j]-1):
					
					data=np.zeros(32,dtype=np.float32)
					for k in range(0,28):
						data[k]=obs_rep[i][k]
					data[28]=paths[i][j][m][0]
					data[29]=paths[i][j][m][1]
					data[30]=paths[i][j][path_lengths[i][j]-1][0]
					data[31]=paths[i][j][path_lengths[i][j]-1][1]
					# print("paths[i][j][m:path_lengths[i][j]]")
					# print(paths[i][j][m:path_lengths[i][j]])
					targets.append(paths[i][j][m+1])
					#new_targets.append(orient_norm[i][j][m])
					#classification_orient_targets.append(orient_classification[i][j][m])
					classification_orient_targets.append(orient_index[i][j][m])
					classification_norm_targets.append(norm[i][j][m])
					dataset.append(data)
					env_indices.append(i)

					all_interpolated_points = []
					if path_lengths[i][j]>2:
						p1 = paths[i][j][m+1] #next point
						for n in range(m+1, path_lengths[i][j]-1):
							p2 = paths[i][j][n+1]
							interpolated_segment = interpolate_points(p1, p2, 0.1) #not include p1 but include p2
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
					
					
	# print("222222")		
	max_length_targets_future_all = max(get_effective_length(arr) for arr in targets_future_all)	
	# Method: Fill each array to make its length equal to the maximum length
	# print("max_length_targets_future_all",max_length_targets_future_all)
	# assert 0
	# print("paths")
	# print(paths)
	# print("targets_future_all")
	# print(targets_future_all)
	# print(len(targets_future_all))
	# print("targets_future_all")
	# print(targets_future_all)
	# assert 0
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
	# print("33333")			
	print("len(padded_targets_future_all)")
	# print(padded_targets_future_all)
	print(len(padded_targets_future_all))
	print("dataset")
	print(len(dataset))
	# print("padded_targets_future_all")
	# print(padded_targets_future_all)
	# print("targets")
	# print(targets)
	# for arr in padded_targets_future_all:
	# 	print("*")
	# 	print(arr)
		
	# 	print(len(arr))
		
	# assert 0
	
	new_dataset=dataset[:]
	orient_dataset=	dataset[:]	

	orient_data=list(zip(env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_targets,classification_norm_targets))
	random.shuffle(orient_data)
	env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_targets,classification_norm_targets=zip(*orient_data)

	# assert 0
	return 	enviro_mask_set,boundary_points_set,boundary_lengths,np.asarray(env_indices),np.asarray(dataset),np.asarray(targets),np.asarray(padded_targets_future_all),np.asarray(orient_dataset),np.asarray(classification_orient_targets),np.asarray(classification_norm_targets) 
	#return 	np.asarray(dataset),np.asarray(targets),np.asarray(new_dataset),np.asarray(new_targets) 







#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000

def load_test_dataset_crossentropy(scale_para,bias,num_sector,N=10, NP=200,s=100, sp=0):	
	
	# obc=np.zeros((N,11,2),dtype=np.float32)
	# temp=np.fromfile('../../DATA_SET/Concave_2D/obs.dat')
	# obs=temp.reshape(len(temp)//2,2) #change by z 

	# temp=np.fromfile('../../DATA_SET/Concave_2D/obs_perm_concave.dat',np.int32)
	# perm=temp.reshape(167960,11)

	# ## loading obstacles
	# for i in range(0,N):
	# 	for j in range(0,11):
	# 		for k in range(0,2):

	# 			obc[i][j][k]=obs[perm[i+s][j]][k]



	# Q = Encoder()
	# Q.load_state_dict(torch.load('./AE_complex/models/concave_used/cae_encoder_400.pkl'))
	# if torch.cuda.is_available():
	# 	Q.cuda()

	## Calculate the longest set of boundary points
	max_boundary_length=0
	boundary_lengths=np.zeros((N),dtype=int)
	for i in range(0,N):
		
		file_path = '../../DATA_SET/Concave_2D/obc_mask_concave_narrow/concave_enviro_' + str(i+s) + '.dat'  # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)

		surface = oh.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh.surface_to_real(surface_points,bias,scale_para) 
		boundary_lengths[i]=len(boundary_points)
		if len(boundary_points)>max_boundary_length:
			max_boundary_length=len(boundary_points)

	boundary_points_set=np.full((N,max_boundary_length,2),100, dtype=np.float32)   ## Take a very far point [100,100] so that it does not affect subsequent operations
	enviro_mask_set=np.zeros((N,int(40*scale_para), int(40*scale_para)))

	obs_rep=np.zeros((N,28),dtype=np.float32)

	for i in range(0,N):
		#load obstacle point cloud
		file_path = '../../DATA_SET/Concave_2D/obc_mask_concave_narrow/concave_enviro_' + str(i+s) + '.dat'  # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)

		enviro_mask_set[i]=enviro
		surface = oh.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh.surface_to_real(surface_points,bias,scale_para)  #numpy.ndarray
		#print("boundary_points")
		#print(boundary_points)
		#print(boundary_points.shape)
		#print(len(boundary_points))
		#print(type(boundary_points))
		#print("max_boundary_length")
		#print(max_boundary_length)
		#plt.scatter(boundary_points[:,0], boundary_points[:,1], c='red', s=1)
		#plt.scatter(temp[:,0], temp[:,1], c='green', s=1)
		#plt.savefig('./point_cloud_img/obc'+str(i+s)+'.png')
		#plt.clf()
		for k in range(0,len(boundary_points)):
			boundary_points_set[i][k]=boundary_points[k]

	
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/Concave_2D/Path_concave_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2) #change by z
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)	
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
	
	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/Concave_2D/Path_concave_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2) #change by z
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
				
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
	
	#orient and norm of the path by z 
	accuracy=num_sector #The number of spatial sectors  if change the value of accuracy, outputsize in model also need to be change
	orient=np.zeros((N,NP,max_length-1),dtype=np.float32)
	norm=np.zeros((N,NP,max_length-1),dtype=np.float32)
	# orient_norm=np.zeros((N,NP,max_length-1,2),dtype=np.float32)
	# orient_360=np.zeros((N,NP,max_length-1),dtype=np.float32)
	orient_classification=np.zeros((N,NP,max_length-1,accuracy),dtype=np.float32)
	orient_index=np.zeros((N,NP,max_length-1),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/Concave_2D/Path_concave_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path_reshape=path.reshape(len(path)//2,2) #change by z
				for k in range(0,len(path_reshape)-1):
					orient[i][j][k]=math.atan2(path_reshape[k+1][1]-path_reshape[k][1],path_reshape[k+1][0]-path_reshape[k][0])/(2*math.pi)*360
					norm[i][j][k]=math.sqrt(math.pow(path_reshape[k+1][1]-path_reshape[k][1],2)+math.pow(path_reshape[k+1][0]-path_reshape[k][0],2))
					
					index=int(orient[i][j][k]//(360/accuracy)+accuracy/2)
					if(index==accuracy):
						index=accuracy-1  #防止orient_classification恰好超出索引
					orient_classification[i][j][k][index]=1.0
					orient_index[i][j][k]=index
						
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
	classification_orient_targets=[]
	classification_norm_targets=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:	

				for m in range(0, path_lengths[i][j]-1):
					
					data=np.zeros(32,dtype=np.float32)
					for k in range(0,28):
						data[k]=obs_rep[i][k]
					data[28]=paths[i][j][m][0]
					data[29]=paths[i][j][m][1]
					data[30]=paths[i][j][path_lengths[i][j]-1][0]
					data[31]=paths[i][j][path_lengths[i][j]-1][1]
						
					targets.append(paths[i][j][m+1])
					# new_targets.append(orient_norm[i][j][m])
					#classification_orient_targets.append(orient_classification[i][j][m])
					classification_orient_targets.append(orient_index[i][j][m])
					classification_norm_targets.append(norm[i][j][m])
					dataset.append(data)
					
				
					#env_indices.append(i+s) #i+s is real environment indices !!!!!!!!!!!!!!!!
					env_indices.append(i) #i+s is real environment indices. obc[i][j][k]=obs[perm[i+s][j]][k] and obc[idx] 
					
					all_interpolated_points = []
					if path_lengths[i][j]>2:
						p1 = paths[i][j][m+1] #next point
						for n in range(m+1, path_lengths[i][j]-1):
							p2 = paths[i][j][n+1]
							interpolated_segment = interpolate_points(p1, p2, 0.1) #not include p1 but include p2
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

	
	
	orient_data=list(zip(env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_targets,classification_norm_targets))
	random.shuffle(orient_data)
	env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_targets,classification_norm_targets=zip(*orient_data)

	return 	enviro_mask_set,boundary_points_set,boundary_lengths,np.asarray(env_indices),np.asarray(dataset),np.asarray(targets),np.asarray(padded_targets_future_all),np.asarray(orient_dataset),np.asarray(classification_orient_targets),np.asarray(classification_norm_targets) 
	 

# #N=number of environments; NP=Number of Paths
# def load_dataset(N=100,NP=4000):

# 	Q = Encoder()
# 	Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
# 	if torch.cuda.is_available():
# 		Q.cuda()

		
# 	obs_rep=np.zeros((N,28),dtype=np.float32)
# 	for i in range(0,N):
# 		#load obstacle point cloud
# 		temp=np.fromfile('../../dataset/obs_cloud_test/obc'+str(i)+'.dat')
# 		temp=temp.reshape(len(temp)/2,2)
# 		obstacles=np.zeros((1,2800),dtype=np.float32)
# 		obstacles[0]=temp.flatten()
# 		inp=torch.from_numpy(obstacles)
# 		inp=Variable(inp).cuda()
# 		output=Q(inp)
# 		output=output.data.cpu()
# 		obs_rep[i]=output.numpy()



	
# 	## calculating length of the longest trajectory
# 	max_length=0
# 	path_lengths=np.zeros((N,NP),dtype=int)
# 	for i in range(0,N):
# 		for j in range(0,NP):
# 			fname='../../dataset/e'+str(i)+'/path'+str(j)+'.dat'
# 			if os.path.isfile(fname):
# 				path=np.fromfile(fname)
# 				path=path.reshape(len(path)/2,2)
# 				path_lengths[i][j]=len(path)	
# 				if len(path)> max_length:
# 					max_length=len(path)
			

# 	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

# 	for i in range(0,N):
# 		for j in range(0,NP):
# 			fname='../../dataset/e'+str(i)+'/path'+str(j)+'.dat'
# 			if os.path.isfile(fname):
# 				path=np.fromfile(fname)
# 				path=path.reshape(len(path)/2,2)
# 				for k in range(0,len(path)):
# 					paths[i][j][k]=path[k]
	
					

# 	dataset=[]
# 	targets=[]
# 	for i in range(0,N):
# 		for j in range(0,NP):
# 			if path_lengths[i][j]>0:				
# 				for m in range(0, path_lengths[i][j]-1):
# 					data=np.zeros(32,dtype=np.float32)
# 					for k in range(0,28):
# 						data[k]=obs_rep[i][k]
# 					data[28]=paths[i][j][m][0]
# 					data[29]=paths[i][j][m][1]
# 					data[30]=paths[i][j][path_lengths[i][j]-1][0]
# 					data[31]=paths[i][j][path_lengths[i][j]-1][1]
						
# 					targets.append(paths[i][j][m+1])
# 					dataset.append(data)
			
# 	data=zip(dataset,targets)
# 	random.shuffle(data)	
# 	dataset,targets=zip(*data)
# 	return 	np.asarray(dataset),np.asarray(targets) 


# #N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
# #Unseen_environments==> N=10, NP=2000,s=100, sp=0
# #seen_environments==> N=100, NP=200,s=0, sp=4000

# def load_test_dataset(N=10, NP=2000,s=100, sp=0): #change by z unseen environments
# #def load_test_dataset(N=100,NP=200, s=0,sp=4000):
# 	obs_start=s
# 	obc=np.zeros((N,11,2),dtype=np.float32)
# 	temp=np.fromfile('./dataset2/obs.dat')
# 	obs=temp.reshape(len(temp)//2,2) #change by z 

# 	temp=np.fromfile('./dataset2/obs_perm2.dat',np.int32)
# 	perm=temp.reshape(167960,11)

# 	## loading obstacles
# 	for i in range(0,N):
# 		for j in range(0,11):
# 			for k in range(0,2):

# 				obc[i][j][k]=obs[perm[i+s][j]][k]
	
					
# 	Q = Encoder()
# 	D = Decoder()
# 	Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
# 	D.load_state_dict(torch.load('../models/cae_decoder.pkl'))
# 	# Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
# 	# D.load_state_dict(torch.load('../models/cae_decoder.pkl'))
# 	if torch.cuda.is_available():
# 		Q.cuda()
# 		D.cuda()
	
# 	obs_rep=np.zeros((N,28),dtype=np.float32)	
# 	obs_recover=np.zeros((N,2800),dtype=np.float32)
# 	k=0
# 	for i in range(s,s+N): #s+N
# 		temp=np.fromfile('./dataset2/obs_cloud_test/obc'+str(i)+'.dat')
# 		temp=temp.reshape(len(temp)//2,2) #change by z 
# 		obstacles=np.zeros((1,2800),dtype=np.float32)
# 		obstacles[0]=temp.flatten()
# 		inp=torch.from_numpy(obstacles)
# 		inp=Variable(inp).cuda()
# 		output=Q(inp)
# 		recover=D(output) #by z
# 		output=output.data.cpu()
# 		recover=recover.data.cpu()#by z
# 		obs_rep[k]=output.numpy()
# 		obs_recover[k]=recover.numpy()#by z

# 		k=k+1
# 	## calculating length of the longest trajectory
# 	max_length=0
# 	path_lengths=np.zeros((N,NP),dtype=int)
# 	for i in range(0,N):
# 		for j in range(0,NP):
# 			fname='./dataset2/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
# 			if os.path.isfile(fname):
# 				path=np.fromfile(fname)
# 				path=path.reshape(len(path)//2,2) #change by z 
# 				path_lengths[i][j]=len(path)	
# 				if len(path)> max_length:
# 					max_length=len(path)
			

# 	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

# 	for i in range(0,N):
# 		for j in range(0,NP):
# 			fname='./dataset2/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
# 			if os.path.isfile(fname):
# 				path=np.fromfile(fname)
# 				path=path.reshape(len(path)//2,2) #change by z 
# 				for k in range(0,len(path)):
# 					paths[i][j][k]=path[k]
	
					



# 	return 	obc,obs_rep,paths,path_lengths,obs_start,obs_recover
	


def load_test_dataset_planner(scale_para,bias,N=10, NP=2000,s=100, sp=0): #change by z unseen environments
#def load_test_dataset_new(scale_para,bias,N=100,NP=200, s=0,sp=4000):'

	obs_start=s
	obc=np.zeros((N,11,2),dtype=np.float32)
	temp=np.fromfile('../../DATA_SET/Concave_2D/obs.dat')
	obs=temp.reshape(len(temp)//2,2) #change by z 

	temp=np.fromfile('../../DATA_SET/Concave_2D/obs_perm_concave.dat',np.int32)
	perm=temp.reshape(167960,11)

	## loading obstacles
	for i in range(0,N):
		for j in range(0,11):
			for k in range(0,2):

				obc[i][j][k]=obs[perm[i+s][j]][k]
	
					
	Q = Encoder()
	D = Decoder()
	Q.load_state_dict(torch.load('./AE_complex/models/cae_encoder_concave_2d_300_average loss_2.102830.pkl'))
	D.load_state_dict(torch.load('./AE_complex/models/cae_decoder_concave_2d_300_average loss_2.102830.pkl'))
	# Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	# D.load_state_dict(torch.load('../models/cae_decoder.pkl'))
	if torch.cuda.is_available():
		Q.cuda()
		D.cuda()
	

	## Calculate the longest set of boundary points
	max_boundary_length=0
	boundary_lengths=np.zeros((N),dtype=int)
	for i in range(0,N):
		# temp=np.fromfile('../../DATA_SET/Concave_2D/obc_cloud_concave/obc_concave'+str(i+s)+'.dat') 
		# temp=temp.reshape(len(temp)//2,2)

		# enviro = np.zeros((int(40*scale_para), int(40*scale_para)))
		# length_obc_mask=length_obc*scale_para
		# width_obc_mask=width_obc*scale_para
		# obc_mask=(obc[i]+bias)*scale_para
		# for j in range(0,11):
		# 	center_x, center_y = obc_mask[j]
		# 	#Calculate the coordinates of the four corners of the obstacle
		# 	left_x = int(np.floor(center_x - length_obc_mask[j] / 2))
		# 	right_x = int(np.ceil(center_x + length_obc_mask[j] / 2))
		# 	top_y = int(np.floor(center_y - width_obc_mask[j] / 2))
		# 	bottom_y = int(np.ceil(center_y + width_obc_mask[j]/ 2))
		# 	for x in range(left_x, right_x + 1):
		# 		for y in range(top_y, bottom_y + 1):
		# 			# if 0 <= x < enviro.shape[0] and 0 <= y < enviro.shape[1]:  # 确保标记不超出地图范围
		# 			enviro[x, y] = 1
		# enviro = ndimage.binary_fill_holes(enviro).astype(int) #mask

		file_path = '../../DATA_SET/Concave_2D/obc_mask_narrow/narrow_enviro_' + str(i+s) + '.dat' # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)


		surface = oh.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh.surface_to_real(surface_points,bias,scale_para) 
		boundary_lengths[i]=len(boundary_points)
		if len(boundary_points)>max_boundary_length:
			max_boundary_length=len(boundary_points)
	

	boundary_points_set=np.full((N,max_boundary_length,2),100, dtype=np.float32)   ## Take a very far point [100,100] so that it does not affect subsequent operations
	enviro_mask_set=np.zeros((N,int(40*scale_para), int(40*scale_para)))


	obs_rep=np.zeros((N,28),dtype=np.float32)	
	obs_recover=np.zeros((N,8800),dtype=np.float32)

	for i in range(0,N): 
		# temp=np.fromfile('../../DATA_SET/Concave_2D/obc_cloud_concave/obc_concave'+str(i+s)+'.dat') #obs_cloud_test not been changed
		# temp=temp.reshape(len(temp)//2,2) #change by z 

		# enviro = np.zeros((int(40*scale_para), int(40*scale_para)))
		# length_obc_mask=length_obc*scale_para
		# width_obc_mask=width_obc*scale_para
		# obc_mask=(obc[i]+bias)*scale_para
		# for j in range(0,11):
		# 	center_x, center_y = obc_mask[j]
		# 	#Calculate the coordinates of the four corners of the obstacle
		# 	left_x = int(np.floor(center_x - length_obc_mask[j] / 2))
		# 	right_x = int(np.ceil(center_x + length_obc_mask[j] / 2))
		# 	top_y = int(np.floor(center_y - width_obc_mask[j] / 2))
		# 	bottom_y = int(np.ceil(center_y + width_obc_mask[j]/ 2))
		# 	for x in range(left_x, right_x + 1):
		# 		for y in range(top_y, bottom_y + 1):
		# 			# if 0 <= x < enviro.shape[0] and 0 <= y < enviro.shape[1]:  # 确保标记不超出地图范围
		# 			enviro[x, y] = 1
		# enviro = ndimage.binary_fill_holes(enviro).astype(int) #mask
		

		file_path = '../../DATA_SET/Concave_2D/obc_mask_narrow/narrow_enviro_' + str(i+s) + '.dat' # 根据需要更改文件名和路径
		shape = (int(40 * scale_para), int(40 * scale_para))
		enviro = np.fromfile(file_path, dtype=int).reshape(shape)

		enviro_mask_set[i]=enviro
		surface = oh.sample_inner_surface_in_pixel(torch.from_numpy(enviro))
		surface = surface.cpu().numpy()
		surface_points = oh.SamplesFunc(surface) #Extract inner surface points
		boundary_points=oh.surface_to_real(surface_points,bias,scale_para)  #numpy.ndarray
		for k in range(0,len(boundary_points)):
			boundary_points_set[i][k]=boundary_points[k]
		
		# obstacles=np.zeros((1,8800),dtype=np.float32)
		# obstacles[0]=temp.flatten()
		# inp=torch.from_numpy(obstacles)
		# inp=Variable(inp).cuda()
		# output=Q(inp)
		# recover=D(output) #by z
		# output=output.data.cpu()
		# recover=recover.data.cpu()#by z
		# obs_rep[i]=output.numpy()
		# obs_recover[i]=recover.numpy()#by z

		
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=int)
	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/Concave_2D/env_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2) #change by z 
				path_lengths[i][j]=len(path)	
				if len(path)> max_length:
					max_length=len(path)
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")

	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname='../../DATA_SET/Concave_2D/env_narrow/e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2) #change by z 
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
			else:
				print("Alert! No relevant files found.")
				print("The file index that cannot be found is e"+str(i+s)+"/path"+str(j+sp)+".dat")
					


	return 	enviro_mask_set,obc,paths,path_lengths,obs_start,obs_recover
	
	# return 	enviro_mask_set,obc,obs_rep,paths,path_lengths,obs_start,obs_recover
	

# obs_cloud_train just used for CAE trainning. do not use it in here
