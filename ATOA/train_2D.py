import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader_crossentropy_concave_2d import load_dataset_crossentropy
from data_loader_crossentropy_concave_2d import load_test_dataset_crossentropy

from model import MLP 
from torch.autograd import Variable 
import math
import time
import AE_concave_2d_CNN.CNN_2d as AE

#Detect whether data explosion has occurred
def check_data_valid(data_list):
	for data in data_list:
		if torch.isnan(data).any():
			print("DISCOVER NAN!")
			print(data_list)
			assert 0
		if torch.isinf(data).any():
			print("DISCOVER INF!")
			print(data_list)
			assert 0

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def get_input(i,data,targets,bs):

	if i+bs<len(data):
		bi=data[i:i+bs]
		bt=targets[i:i+bs]	
	else:
		bi=data[i:]
		bt=targets[i:]
		
	return torch.from_numpy(bi),torch.from_numpy(bt)

def get_input_orient(i,env_indices,targets,padded_targets_future_all,data,classification_orient_targets,classification_norm_targets,bs):

	if i+bs<len(data):
		bi=data[i:i+bs]
		bt1=classification_norm_targets[i:i+bs]
		bt2=classification_orient_targets[i:i+bs]	
		current_point=bi[:,28:30]
		target_point=targets[i:i+bs]	
		targets_point_future_all_batch=padded_targets_future_all[i:i+bs]
		env_indices_batch=env_indices[i:i+bs]
	

	else:
		bi=data[i:]
		bt1=classification_norm_targets[i:]
		bt2=classification_orient_targets[i:] #orient_label
		current_point=bi[:,28:30]
		target_point=targets[i:i+bs]
		targets_point_future_all_batch=padded_targets_future_all[i:i+bs]
		env_indices_batch=env_indices[i:]
	
		
	return torch.from_numpy(env_indices_batch),torch.from_numpy(bi),torch.from_numpy(bt1),torch.from_numpy(bt2),torch.from_numpy(current_point),torch.from_numpy(target_point),targets_point_future_all_batch



mse_loss= nn.MSELoss(reduction='mean')
mse_loss_test= nn.MSELoss(reduction='mean')

Cross_loss = nn.CrossEntropyLoss()
Cross_loss_test = nn.CrossEntropyLoss()

def new_norm_loss(x, a, b, c):
    sqrt_2 = torch.sqrt(torch.tensor(2.0)) 
    return  (a * torch.exp(-b * x) + c * torch.log(1 + torch.exp(x - 40 * sqrt_2)))

# def norm_loss_relu(magnitude, target_magnitude=25, scale=0.1):
#     loss = scale * (target_magnitude - magnitude)
#     # return loss
#     return torch.relu(loss)  # 使用PyTorch的ReLU以支持反向传播

def norm_loss_relu(x,distance_left):
    epsilon = 1e-10
    distance_left = torch.clamp(distance_left, min=epsilon)
    ratio = torch.clamp(x / distance_left,min=-5) - 1 #if x / distance_left too small, like -10, the exp(100) will become inf
    scaled_ratio = -10 * ratio
    return 0.5 * torch.log1p(torch.exp(scaled_ratio)) #log1p(x)=log(1+x)

def max_out_softm_to_angle(num_sector,out_orient_softm):
	
	max_possibility, max_index = torch.max(out_orient_softm, dim=-1)

	center_angle = np.linspace(-180+360/args.num_sector/2, 180-360/args.num_sector/2, args.num_sector) #center_angle for each sector
	center_angle = torch.from_numpy(center_angle).float().cuda()
	
    # A model that take the most probable angle
	out_angle_max=np.zeros((len(out_orient_softm)),dtype=np.float32)
	out_angle_max = torch.from_numpy(out_angle_max).float().cuda()
	for i in range(0,len(out_orient_softm)):
		out_angle_max[i]=center_angle[max_index[i]]
	out_angle_max=out_angle_max.unsqueeze(1)


	# A model that considers all angle possibilities
	center_angle = center_angle.unsqueeze(0) #cuda
	out_angle = torch.mul(center_angle, out_orient_softm) # (batch_size,num_sector)
	out_angle = torch.sum(out_angle, dim=-1) # (batch_size,)
	out_angle=out_angle.unsqueeze(1) #add a dimension
	

	return out_angle_max

def out_softm_to_angle(num_sector,out_orient_softm):

	center_angle = np.linspace(-180+360/args.num_sector/2, 180-360/args.num_sector/2, args.num_sector) #center_angle for each sector
	center_angle = torch.from_numpy(center_angle).float().cuda()
	center_angle = center_angle.unsqueeze(0) #cuda

	out_angle = torch.mul(center_angle, out_orient_softm) # (batch_size,num_sector)
	
	out_angle = torch.sum(out_angle, dim=-1) # (batch_size,)
	
	out_angle=out_angle.unsqueeze(1) #add a dimension



	return out_angle

def angle_norm_to_coordinate(out_norm,out_angle,bi):
	

	x=bi[:,28].unsqueeze(1)+torch.cos(out_angle/360*2*math.pi)*out_norm
	y=bi[:,29].unsqueeze(1)+torch.sin(out_angle/360*2*math.pi)*out_norm
	
	predict_point = torch.cat((x, y), dim=1)

		
	return predict_point

# def f_DL(x, k):
# 	#return np.exp(-k*(x-3.53553390)) #circumscribed circle
#     return torch.exp(-k*(x-2.5)) #inscribed circle
def f_DL2(x, k):

	return torch.exp(-k*(x)) 

# def Count_Distance_loss(x,idx,obc):  #it is for predict point
	
# 	t=obc[idx].transpose(1,0)
	
# 	x = torch.unsqueeze(x, dim=0)

# 	distance_point_to_obstacle=torch.sqrt(torch.sum(torch.square(torch.sub(t,x)),dim=-1))

# 	distance_point_to_obstacle=f_DL(distance_point_to_obstacle,4)

# 	distance_point_to_obstacle=torch.sum(distance_point_to_obstacle,dim=0)

# 	distance_point_to_obstacle=torch.mean(distance_point_to_obstacle)

# 	return distance_point_to_obstacle

def Predict_point_to_GT_loss(targets_point_future_all_batch,predict_point):
	predict_point_mid=torch.unsqueeze(predict_point, dim=1) #[batch,1,dimension]
	distance_point_to_all_gt=torch.sqrt(torch.sum(torch.square(torch.sub(predict_point_mid,targets_point_future_all_batch)),dim=-1))#[batch,max_future_all_length]
	check_data_valid([distance_point_to_all_gt])
	distance_point_to_gt,index_1=torch.min(distance_point_to_all_gt,dim=-1) #[batch]
	distance_point_to_gt_mean=torch.mean(distance_point_to_gt)

	# print("targets_point_future_all_batch")
	# print(targets_point_future_all_batch)
	# print(targets_point_future_all_batch.shape)
	# print("predict_point")
	# print(predict_point)
	# print(predict_point.shape)
	# print("predict_point_mid")
	# print(predict_point_mid)
	# print(predict_point_mid.shape)
	# print("distance_point_to_all_gt")
	# print(distance_point_to_all_gt)
	# print(distance_point_to_all_gt.shape)
	# print("distance_point_to_gt")
	# print(distance_point_to_gt)
	# print(distance_point_to_gt.shape)
	# print("distance_point_to_gt_mean")
	# print(distance_point_to_gt_mean)
	# print(distance_point_to_gt_mean.shape)
	
	# assert 0

	return distance_point_to_gt_mean

def Count_Distance_mid_points_loss(enviro_mask_set,boundary_points_set,boundary_lengths,predict_point,current_point,idx):  #it is for interpolation point and predicted point
	
	#!!!Please be aware not to use args.batch_size_train to create the dimensions of arrays, as errors may occur when the data is insufficient to fill args.batch_size_train.!!!
	#idx:(batch_size_train); enviro_mask_set:(load_data_N,140,140);	enviro_mask_set[idx]:(batch_size,140,140); 
 	#mask_total_point:[1,num,batch_size_train,2];boundary_points_set:(load_data_N,max_boundary_length,2)
	num=args.interpolation_point_num #interpolation points: interpolation_point_num-1. when num=interpolation_point_num, it represent the predicted point
	total_point=torch.zeros((num,predict_point.shape[0],predict_point.shape[1]))
	total_point=total_point.cuda()
	

	for i in range (0,num):
		total_point[i]=current_point+(predict_point-current_point)*(i+1)/num


	total_point = torch.unsqueeze(total_point, dim=0) #[1,num,batch_size,2]
	# print("total_point")
	# print(total_point)
	check_data_valid([total_point])

	

	boundary_points=boundary_points_set[idx] #(batch_size,max_boundary_length,2)
	boundary_points=to_var(boundary_points)

	boundary_points=boundary_points.transpose(1,0) #(max_boundary_length,batch_size,2)
	boundary_points=torch.unsqueeze(boundary_points, dim=1) #(max_boundary_length,1,batch_size,2)
	# print("boundary_points")
	# print(boundary_points)
	check_data_valid([boundary_points])


	mask_total_point=torch.add(total_point,args.bias)*args.scale_para #mask_total_point:[1,num,batch_size_train,2] [1,5,6,2]
	# print("mask_total_point")
	# print(mask_total_point)
	check_data_valid([mask_total_point])


	mask_total_point_floor=torch.floor(mask_total_point).to(torch.long)
	# mask_total_point_ceil=torch.ceil(mask_total_point).to(torch.long)
	# print("mask_total_point_floor")
	# print(mask_total_point_floor)
	check_data_valid([mask_total_point_floor])


	sign_value=torch.zeros(mask_total_point_floor.shape[2],mask_total_point_floor.shape[1]).cuda()#(batch_size,num)
	
	# T2=time.time()	

	# for i in range(0,mask_total_point_floor.shape[2]):
	# 	for j in range(0,mask_total_point_floor.shape[1]):
	# 		if mask_total_point_floor[0,j,i,0]<enviro_mask_set[idx][i,:,:].shape[0] and mask_total_point_floor[0,j,i,0]>-enviro_mask_set[idx][i,:,:].shape[0] and mask_total_point_floor[0,j,i,1]<enviro_mask_set[idx][i,:,:].shape[1] and mask_total_point_floor[0,j,i,1]>-enviro_mask_set[idx][i,:,:].shape[1]:
	# 			if enviro_mask_set[idx][i,:,:][mask_total_point_floor[0,j,i,0],mask_total_point_floor[0,j,i,1]]==0:
	# 				sign_value[i,j]=1
	# 			if enviro_mask_set[idx][i,:,:][mask_total_point_floor[0,j,i,0],mask_total_point_floor[0,j,i,1]]==1:
	# 				sign_value[i,j]=-1
	# 		else:
	# 			sign_value[i,j]=1
	# T3=time.time()	
	# print("interval")	
	# print(T3-T2)

	#The region [0, 139]-[139, 139] and [139, 0]-[139, 139] of the mask graph has a constant value of 0. 

	temp=mask_total_point_floor>torch.tensor(args.scale_para*40-1)
	temp1=mask_total_point_floor<torch.tensor(0)
	mask_total_point_floor[temp]=args.scale_para*40-1
	mask_total_point_floor[temp1]=args.scale_para*40-1	

	# print("mask_total_point_floor2")
	# print(mask_total_point_floor)
	check_data_valid([mask_total_point_floor])
	

	for i in range(0,mask_total_point_floor.shape[2]):
		sign_value[i,:]=enviro_mask_set[idx][i,:,:][mask_total_point_floor[0,:,i,0],mask_total_point_floor[0,:,i,1]]
	sign_value=torch.logical_not(sign_value).int()-sign_value
	# print("sign_value")
	# print(sign_value)
	# print(sign_value.shape)		
	check_data_valid([sign_value])
	
	
	# assert 0

	sign_value=sign_value.transpose(0,1) #sign_value(num,batch_size)


	#boundary_points(max_boundary_length,1,batch_size,2) total_point[1,num,batch_size,2] #sign_value(num,batch_size)
	distance_point_to_boundary_points=torch.sqrt(torch.sum(torch.square(torch.sub(boundary_points,total_point)),dim=-1)) #(max_boundary_length,num,batch_size)
	# print("distance_point_to_boundary_points")
	# print(distance_point_to_boundary_points)
	# print(distance_point_to_boundary_points.shape)		
	check_data_valid([distance_point_to_boundary_points])

	distance_point_to_obstacle,to_obstacle_index=torch.min(distance_point_to_boundary_points,dim=0) #(num,batch_size)
	# print("distance_point_to_obstacle")
	# print(distance_point_to_obstacle)
	check_data_valid([distance_point_to_obstacle])

	sign_distance_point_to_obstacle=distance_point_to_obstacle*sign_value #(num,batch_size)
	# print("sign_distance_point_to_obstacle")
	# print(sign_distance_point_to_obstacle)
	check_data_valid([sign_distance_point_to_obstacle])

	sign_loss_point_to_obstacle=f_DL2(sign_distance_point_to_obstacle,args.ek) #(num,batch_size)
	# print("sign_loss_point_to_obstacle")
	# print(sign_loss_point_to_obstacle)
	check_data_valid([sign_loss_point_to_obstacle])

	sign_mean_loss_point_to_obstacle=torch.mean(sign_loss_point_to_obstacle,dim=0)#(batch_size)
	# print("sign_mean_loss_point_to_obstacle")
	# print(sign_mean_loss_point_to_obstacle)
	check_data_valid([sign_mean_loss_point_to_obstacle])

	sign_mean_batch=torch.mean(sign_mean_loss_point_to_obstacle)
	# print("sign_mean_batch")
	# print(sign_mean_batch)
	check_data_valid([sign_mean_batch])


	# #_______________________
	# t=obc[idx].transpose(1,0)  #[7,batch_size,2]
	# t = torch.unsqueeze(t, dim=1) #[7,1,batch_size,2]   
	# distance_point_to_obstacle_1=torch.sqrt(torch.sum(torch.square(torch.sub(t,total_point)),dim=-1))  #[7,num,batch_size]
	# line_to_obs_min_collision, min_index = torch.min(distance_point_to_obstacle_1, dim=1) #shortest distance from the line segment to the obstacle #[7,batch_size]
	# # print("line_to_obs_min_collision0.shape")
	# # print(line_to_obs_min_collision.shape)
	# # print(line_to_obs_min_collision)

	# line_to_obs_min_collision,min_index=torch.min(line_to_obs_min_collision,dim=0) #[batch_size]
	# k=torch.sub(line_to_obs_min_collision,2.5)
	# print("k.shape")
	# print(k.shape)
	# print(k)
	# # print("min_index.shape")
	# # print(min_index.shape)
	# # print(min_index)
	# line_to_obs_min_collision=f_DL(line_to_obs_min_collision,4)
	# print("kloss.shape")
	# print(line_to_obs_min_collision.shape)
	# print(line_to_obs_min_collision)
	# line_to_obs_min_collision=torch.mean(line_to_obs_min_collision,dim=0)
	# print("kmean.shape")
	# print(line_to_obs_min_collision.shape)
	# print(line_to_obs_min_collision)
	# print("______________________________________________________________________")

	# if sign_mean_batch>10000:
	# 	assert 0
	return sign_mean_batch



def loss_function(enviro_mask_set,boundary_points_set,boundary_lengths,env_indices_batch,\
		      out_norm,out_orient_raw,bt1,bt2,predict_point,current_point,targets_point_future_all_batch,distance_current_to_goal):
	k1=args.k1
	k2=args.k2
	k3=args.k3
	k4=args.k4

	pp_to_gt_loss=Predict_point_to_GT_loss(targets_point_future_all_batch,predict_point)
	# print("current_point")
	# print(current_point)
	# print(current_point.shape)
	# print("targets_point_future_all_batch")
	# print(targets_point_future_all_batch)
	# print(targets_point_future_all_batch.shape)
	# print("predict_point")
	# print(predict_point)
	# print(predict_point.shape)

	# print("pp_to_gt_loss")
	# print(pp_to_gt_loss)	
	# assert 0
	check_data_valid([pp_to_gt_loss])
	# print("predict_point")
	# print(predict_point)
	# print(predict_point.shape)
	# print("targets_point_future_all_batch")
	# # print(targets_point_future_all_batch)
	# print(targets_point_future_all_batch.shape)
	# assert 0

	# Distance_point_loss=Count_Distance_loss(predict_point,env_indices_batch,obc)
	line_to_obs_collision=Count_Distance_mid_points_loss(enviro_mask_set,boundary_points_set,boundary_lengths,predict_point,current_point,env_indices_batch)
	# print("line_to_obs_collision")
	# print(line_to_obs_collision)	
	check_data_valid([line_to_obs_collision])


	# out_norm_loss=new_norm_loss(out_norm,1431.23,0.1,1)
	# values = torch.tensor([[-5.0], [-10.0], [-20.0]]).cuda()
	out_norm_loss=k1*norm_loss_relu(out_norm,distance_current_to_goal.unsqueeze(1))

	# print("out_norm")
	# print(out_norm)
	# print(out_norm.shape)
	# print("distance_current_to_goal.unsqueeze(1)")
	# print(distance_current_to_goal.unsqueeze(1))
	# print(distance_current_to_goal.unsqueeze(1).shape)
	# print("out_norm_loss")
	# print(out_norm_loss)
	# print(out_norm_loss.shape)
	# assert 0
	

	out_norm_loss=torch.mean(out_norm_loss)
	# out_norm_mse=mse_loss(out_norm,bt1)
	# print("out_norm_loss_mean")
	# print(out_norm_loss)
	# assert 0
	check_data_valid([out_norm_loss])

	loss_cross=Cross_loss(out_orient_raw,bt2)
	# print("loss_cross")
	# print(loss_cross)
	check_data_valid([loss_cross])
	
	loss_all=out_norm_loss+k2*loss_cross+k3*line_to_obs_collision+k4*pp_to_gt_loss
	# loss_all=out_norm_loss+k2*loss_cross+k4*pp_to_gt_loss
	# print("loss_all")
	# print(loss_all)
	check_data_valid([loss_all])
	
	
	# return loss_all,out_norm_loss,loss_cross,line_to_obs_collision,pp_to_gt_loss
	return loss_all,out_norm_loss,k2*loss_cross,k3*line_to_obs_collision,k4*pp_to_gt_loss

def Error_function(Tenviro_mask_set,Tboundary_points_set,Tboundary_lengths,Tenv_indices_batch,\
		   Tout_norm, Tout_orient_raw,Tbt1,Tbt2,Tpredict_point,Tcurrent_point,Ttargets_point_future_all_batch,Tdistance_current_to_goal):

		
	# Tk1=torch.tensor(1.0).cuda()
	# Tk2=torch.tensor(1.0).cuda()
	# Tk3=torch.tensor(0.001).cuda()
	Tk1=args.Tk1
	Tk2=args.Tk2
	Tk3=args.Tk3
	Tk4=args.Tk4



	Ttargets_point_future_all_batch=torch.from_numpy(Ttargets_point_future_all_batch) #class torch.Tensor
	Ttargets_point_future_all_batch=to_var(Ttargets_point_future_all_batch) #device cuda
	Tpp_to_gt_error=Predict_point_to_GT_loss(Ttargets_point_future_all_batch,Tpredict_point)
	

	# TDistance_point_loss=Count_Distance_loss(Tpredict_point,Tenv_indices_batch,Tobc)
	Tline_to_obs_collision=Count_Distance_mid_points_loss(Tenviro_mask_set,Tboundary_points_set,Tboundary_lengths,Tpredict_point,Tcurrent_point,Tenv_indices_batch)

	Tout_norm_error=Tk1*norm_loss_relu(Tout_norm,Tdistance_current_to_goal.unsqueeze(1))

	Tout_norm_error=torch.mean(Tout_norm_error)

	Tcross_error=Cross_loss_test(Tout_orient_raw,Tbt2)
	

	Terror_all=Tout_norm_error+Tk2*Tcross_error+Tk3*Tline_to_obs_collision+Tk4*Tpp_to_gt_error
	
	# TPre_point_error=torch.square(torch.sub(Tpredict_point,Ttarget_point))	
	# TPre_point_error=torch.sqrt(torch.sum(TPre_point_error,dim=1)) 
	# TPre_point_error=torch.mean(TPre_point_error)
	
	return Terror_all,Tout_norm_error,Tk2*Tcross_error,Tk3*Tline_to_obs_collision,Tk4*Tpp_to_gt_error


def error_count(predict_point,target_point):

	Pre_point_error=torch.square(torch.sub(predict_point,target_point))
	
	Pre_point_error=torch.sqrt(torch.sum(Pre_point_error,dim=1)) 
	Pre_point_error=torch.mean(Pre_point_error)


	return Pre_point_error

def Train(mlp,encoder,enviro_mask_set,boundary_points_set,boundary_lengths,env_indices,targets,padded_targets_future_all,orient_dataset,classification_orient_targets,classification_norm_targets,optimizer,total_loss):
	encoder.train()
	mlp.train()	
	loss = 0
	# print("args.batch_size_train")
	# print(args.batch_size_train)
	avg_loss=0
	avg_mse=0
	avg_cross=0
	avg_point_error=0
	avg_line_to_obs_loss=0
	avg_pp_to_gt_loss=0
	avg_out_norm_loss=0

	print("enviro_mask_set",enviro_mask_set.shape)
	
	for i in range (0,len(orient_dataset),args.batch_size_train):  #step=batch_size_train 
		# Forward, Backward and Optimize
		# encoder.zero_grad()
		# mlp.zero_grad()	
		optimizer.zero_grad()
				
		env_indices_batch,bi,bt1,bt2,current_point,target_point,targets_point_future_all_batch= get_input_orient(i,env_indices,targets,padded_targets_future_all,orient_dataset,classification_orient_targets,classification_norm_targets,args.batch_size_train)
		targets_point_future_all_batch=to_var(targets_point_future_all_batch)
		# print("env_indices_batch")
		# print(env_indices_batch)
		# print(env_indices_batch.shape)
		# print("enviro_mask_set")
		# print(enviro_mask_set[env_indices_batch].shape)
	
		# print("targets_point_future_all_batch")
		# print(targets_point_future_all_batch)
		# print(targets_point_future_all_batch.shape)
		#target_point=to_var(target_point)
		check_data_valid([env_indices_batch,bi,bt1,bt2,current_point,target_point,targets_point_future_all_batch])

		bi=to_var(bi)   #[batch_size, 32]
		
		distance_current_to_goal=torch.sqrt(torch.square(bi[:,30]-bi[:,28])+torch.square(bi[:,31]-bi[:,29]))#[batch_size]

		bt1=to_var(bt1)    
		env_indices_batch=to_var(env_indices_batch)
		bt1 = torch.unsqueeze(bt1, dim=1) #from [batch_size] to [batch_size,1]
		bt2=to_var(bt2)
		# print("env_indices_batch")
		# print(env_indices_batch)    
		# print(env_indices_batch.shape) 
		# print("bi")
		# print(bi)    
		# print(bi.shape)   		
		# print("bt1")
		# print(bt1)    
		# print(bt1.shape)  
		# print("bt2")
		# print(bt2)    
		# print(bt2.shape)  
		# assert 0	

		zn=encoder(enviro_mask_set[env_indices_batch].unsqueeze(1))
		mlp_in = torch.cat((zn,bi[:,28:]), 1)    # keep the first dim the same (# samples) [path_length-1,32]
		# print("zn")
		# print(zn)
		# print(zn.shape)
		# print("bi")
		# print(bi)
		# print(bi.shape)
		# print("mlp_in")
		# print(mlp_in)
		# print(mlp_in.shape)
		
		out_norm, out_orient_raw, out_orient_softm = mlp(mlp_in)   #torch.Tensor+cuda
		
		check_data_valid([out_norm,out_orient_raw,out_orient_softm])
		
		out_angle=out_softm_to_angle(args.num_sector,out_orient_softm)
		
		predict_point=angle_norm_to_coordinate(out_norm,out_angle,bi) #cuda
		
		current_point=to_var(current_point)

		check_data_valid([out_angle,predict_point,current_point])
		
		loss_all,out_norm_loss,loss_cross,line_to_obs_loss,pp_to_gt_loss = loss_function(enviro_mask_set,boundary_points_set,boundary_lengths,env_indices_batch,\
							   out_norm,out_orient_raw,bt1,bt2,predict_point,current_point,targets_point_future_all_batch,distance_current_to_goal) #1

		

		loss_all.backward()
		optimizer.step()

		predict_point_cpu=predict_point.cpu()
		Pre_point_error=error_count(predict_point_cpu,target_point)
		avg_loss=avg_loss+loss_all.data #change by z
		avg_out_norm_loss=avg_out_norm_loss+out_norm_loss.data #change by z
		avg_cross=avg_cross+loss_cross.data #change by z
		avg_line_to_obs_loss=avg_line_to_obs_loss+line_to_obs_loss
		avg_pp_to_gt_loss=avg_pp_to_gt_loss+pp_to_gt_loss
		# avg_point_error=avg_point_error+Pre_point_error.data
	
	print ("load_NP:",args.load_data_NP,"dropout:",args.dropout_p_MLP) #change by z
	print ("--average loss:") #change by z
	print (avg_loss/math.ceil(len(orient_dataset)/args.batch_size_train))#change by z
	print ("--average out_norm_loss:") #change by z
	print (avg_out_norm_loss/math.ceil(len(orient_dataset)/args.batch_size_train))#change by z
	print ("--average cross_loss:") #change by z
	print (avg_cross/math.ceil(len(orient_dataset)/args.batch_size_train))#change by z
	print ("--average line_to_obs_loss:") #change by z
	print (avg_line_to_obs_loss/math.ceil(len(orient_dataset)/args.batch_size_train))#change by z
	print ("--average pp_to_gt_loss:") #change by z
	print (avg_pp_to_gt_loss/math.ceil(len(orient_dataset)/args.batch_size_train))#change by z
	# print ("--average point_error:") #change by z
	# print (avg_point_error/math.ceil(len(orient_dataset)/args.batch_size_train))#change by z

	total_loss.append(avg_loss/(len(orient_dataset)/args.batch_size_train))

	return total_loss,avg_loss,avg_out_norm_loss,avg_cross,avg_line_to_obs_loss,avg_pp_to_gt_loss,avg_point_error



def Test(mlp,encoder,Tenviro_mask_set,Tboundary_points_set,Tboundary_lengths,Tenv_indices,Ttargets,Tpadded_targets_future_all,Torient_dataset,Tclassification_orient_targets,Tclassification_norm_targets,Ttotal_error):
	mlp.eval()
	encoder.eval()
	Tavg_error=0.0
	Tavg_out_norm_error=0.0
	Tavg_cross_error=0.0
	Tavg_line_to_obs_collision=0.0
	Tavg_Pre_point_error=0.0
	Tavg_pp_to_gt_error=0
	

	Tboundary_points_set=torch.from_numpy(Tboundary_points_set) #class torch.Tensor
	Tboundary_points_set=to_var(Tboundary_points_set) #device cuda
	Tboundary_lengths=torch.from_numpy(Tboundary_lengths) #class torch.Tensor
	Tboundary_lengths=to_var(Tboundary_lengths) #device cuda
	Tenviro_mask_set=torch.from_numpy(Tenviro_mask_set).float() #class torch.Tensor
	Tenviro_mask_set=to_var(Tenviro_mask_set) #device cuda


	with torch.no_grad():
		for i in range (0,len(Torient_dataset),args.batch_size_test):  #step=batch_size_test
			# Forward 
			Tenv_indices_batch,Tbi,Tbt1,Tbt2,Tcurrent_point,Ttarget_point,Ttargets_point_future_all_batch= get_input_orient(i,Tenv_indices,Ttargets,Tpadded_targets_future_all,Torient_dataset,Tclassification_orient_targets,Tclassification_norm_targets,args.batch_size_test)
			
			Ttarget_point=to_var(Ttarget_point)
			Tbi=to_var(Tbi)   #[batch_size_test, 2]
			Tdistance_current_to_goal=torch.sqrt(torch.square(Tbi[:,30]-Tbi[:,28])+torch.square(Tbi[:,31]-Tbi[:,29]))#[batch_size]


			Tbt1=to_var(Tbt1)    
			Tenv_indices_batch=to_var(Tenv_indices_batch)
			Tbt1 = torch.unsqueeze(Tbt1, dim=1) #from [batch_size_test] to [batch_size_test,1]
			Tbt2=to_var(Tbt2)

			Tzn=encoder(Tenviro_mask_set[Tenv_indices_batch].unsqueeze(1))
			Tmlp_in = torch.cat((Tzn,Tbi[:,28:]), 1)    # keep the first dim the same (# samples) [path_length-1,32]
		
			Tout_norm, Tout_orient_raw, Tout_orient_softm = mlp(Tmlp_in)   #([batch_size_test, 2])

			#Tout_angle=out_softm_to_angle(args.num_sector,Tout_orient_softm) # A model that considers all angle possibilities
			Tout_angle=max_out_softm_to_angle(args.num_sector,Tout_orient_softm) # A model that take the most probable angle

			Tpredict_point=angle_norm_to_coordinate(Tout_norm,Tout_angle,Tbi)
			
			Tcurrent_point=to_var(Tcurrent_point)

			Terror_all,Tout_norm_error,Tcross_error,Tline_to_obs_collision,Tpp_to_gt_error= Error_function(Tenviro_mask_set,Tboundary_points_set,Tboundary_lengths,Tenv_indices_batch,Tout_norm, Tout_orient_raw,Tbt1,Tbt2,Tpredict_point,Tcurrent_point,Ttargets_point_future_all_batch,Tdistance_current_to_goal) #1
			
			
			Tavg_error=Tavg_error+Terror_all.data #change by z
			Tavg_out_norm_error=Tavg_out_norm_error+Tout_norm_error.data #change by z
			Tavg_cross_error=Tavg_cross_error+Tcross_error.data #change by z
			Tavg_line_to_obs_collision=Tavg_line_to_obs_collision+Tline_to_obs_collision
			Tavg_pp_to_gt_error=Tavg_pp_to_gt_error+Tpp_to_gt_error
			# Tavg_Pre_point_error=Tavg_Pre_point_error+TPre_point_error.data

		
				
	Ttotal_error.append(Tavg_error/(len(Torient_dataset)/args.batch_size_test))	
	return Ttotal_error,Tavg_error,Tavg_out_norm_error,Tavg_cross_error,Tavg_line_to_obs_collision,Tavg_pp_to_gt_error,Tavg_Pre_point_error
	# return test_norm_error, test_angle_error, test_angle_accuracy, test_point_error



def main(args):
	# Create model directory
 
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
    
	# Build data loader
	#N=number of environments; NP=Number of Paths default:N=100,NP=4000
	enviro_mask_set,boundary_points_set,boundary_lengths,env_indices,dataset,targets,padded_targets_future_all,orient_dataset,classification_orient_targets,classification_norm_targets= load_dataset_crossentropy(scale_para=args.scale_para,bias=args.bias,num_sector=args.num_sector,N=args.load_data_N,NP=args.load_data_NP) #type: numpy.ndarray N=10,NP=400
	# assert 0
	print("len(targets)",len(targets))
	print("len(classification_orient_targets)",classification_orient_targets,len(classification_orient_targets))

	# assert 0
	
	#Unseen_environments==> N=10, NP=2000,s=100, sp=0
	#seen_environments==> N=100, NP=200,s=0, sp=4000
	#Seen environment
	if args.seen_environment_test:
		Tenviro_mask_set,Tboundary_points_set,Tboundary_lengths,Tenv_indices,Tdataset,Ttargets,Tpadded_targets_future_all,Torient_dataset,Tclassification_orient_targets,Tclassification_norm_targets= load_test_dataset_crossentropy(scale_para=args.scale_para,bias=args.bias,num_sector=args.num_sector,N=args.T_data_N,NP=args.T_data_NP,s=args.T_data_s,sp=args.T_data_sp) #type: numpy.ndarray N=100,NP=50,s=0,sp=4000
		# print("len(Ttargets)",len(Ttargets))
		# print("len(Tclassification_orient_targets)",Tclassification_orient_targets,len(Tclassification_orient_targets))
		# assert 0
		# Tobc=torch.from_numpy(Tobc) #class torch.Tensor
		# Tobc=to_var(Tobc) #device cuda
		

	#Unseen environment
	if args.Unseen_environment_test:
		UTenviro_mask_set,UTboundary_points_set,UTboundary_lengths,UTenv_indices,UTdataset,UTtargets,UTpadded_targets_future_all,UTorient_dataset,UTclassification_orient_targets,UTclassification_norm_targets= load_test_dataset_crossentropy(scale_para=args.scale_para,bias=args.bias,num_sector=args.num_sector,N=args.UT_data_N, NP=args.UT_data_NP,s=args.UT_data_s, sp=args.UT_data_sp) #type: numpy.ndarray N=10, NP=500,s=100, sp=0
		print("Unseen len(targets)")
		print(len(UTtargets))

		# UTobc=torch.from_numpy(UTobc) #class torch.Tensor
		# UTobc=to_var(UTobc) #device cuda

		# assert 0
	
	

	# obc=torch.from_numpy(obc) #class torch.Tensor
	# obc=to_var(obc) #device cuda
	boundary_points_set=torch.from_numpy(boundary_points_set) #class torch.Tensor
	# boundary_points_set=to_var(boundary_points_set) #device cuda
	boundary_lengths=torch.from_numpy(boundary_lengths) #class torch.Tensor
	# boundary_lengths=to_var(boundary_lengths) #device cuda
	enviro_mask_set=torch.from_numpy(enviro_mask_set).float() #class torch.Tensor  float64 to float32
	enviro_mask_set=to_var(enviro_mask_set) #device cuda
	padded_targets_future_all=torch.from_numpy(padded_targets_future_all) #class torch.Tensor
	# padded_targets_future_all=to_var(padded_targets_future_all) #device cuda
	


	# Build the models
	mlp = MLP(args.input_size,args.num_sector,args.dropout_p_MLP)
	encoder=AE.Encoder_CNN_2D(int(args.scale_para*40),args.dropout_p_CNN)
	if args.preload_all:
		mlp.load_state_dict(torch.load('models/crossentropy/test/CNN_NEWLOSS/MLP_E2E_cross_mask_concave2d_epoch=4_N_10_Np_400_avg_loss=16.299_avg_out_norm_loss=0.416_avg_cross=14.632_avg_line_to_obs_loss=0.159_avg_pp_to_gt_loss=1.092_avg_point_error=10.279.pkl'))
		print("load_the_pre-trained_mlp_model")
		encoder.load_state_dict(torch.load('models/crossentropy/test/CNN_NEWLOSS/CNN_E2E_cross_mask_concave2d_epoch=4_N_10_Np_400_avg_loss=16.299_avg_out_norm_loss=0.416_avg_cross=14.632_avg_line_to_obs_loss=0.159_avg_pp_to_gt_loss=1.092_avg_point_error=10.279.pkl'))
		print("load_the_pre-trained_encoder_model")
	elif args.preload_encoder:
		encoder.load_state_dict(torch.load('models/crossentropy/test/CNN_NEWLOSS/Encoder_model/cae_encoder_cnn_2d_epoch_400_average loss_0.008726_Validation average loss_0.000000.pkl'))
		print("just load_the_pre-trained_encoder_model")
		
	if torch.cuda.is_available():
		mlp.cuda()
		encoder.cuda()
	# assert 0

	#Optimizer
	
	# optimizer = torch.optim.Adam(list(encoder.parameters())+list(mlp.parameters()),lr=args.learning_rate) 
	# 定义优化器，为encoder和mlp设置不同的学习率
	if args.preload_encoder or args.preload_all:
		optimizer = torch.optim.Adam([
			{'params': encoder.parameters(), 'lr': args.learning_rate_encoder},  # 为encoder设置的学习率
			{'params': mlp.parameters(), 'lr': args.learning_rate_planner}       # 为mlp设置的学习率
		])
		
	else:
		optimizer = torch.optim.Adam(list(encoder.parameters())+list(mlp.parameters()),lr=args.learning_rate_together) 

	#optimizer = torch.optim.Adagrad(mlp.parameters()) 
    
	# Train the Models
	total_loss=[]
	Ttotal_error=[]
	UTtotal_error=[]

	sm=50 # start saving models after 100 epochs
	for epoch in range(args.num_epochs):
		print ("epoch" + str(epoch)+"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") #change by z
		avg_loss=0.0
		avg_mse=0.0
		avg_out_norm_loss=0.0
		avg_cross=0.0
		avg_line_to_obs_loss=0.0
		avg_pp_to_gt_loss=0.0
		avg_point_error=0.0
		
	
		# remember to revise the dataset size in data_loader file, both train and test
  
		total_loss,avg_loss,avg_out_norm_loss,avg_cross,avg_line_to_obs_loss,avg_pp_to_gt_loss,avg_point_error=Train(mlp,encoder,enviro_mask_set,boundary_points_set,boundary_lengths,env_indices,targets,padded_targets_future_all,orient_dataset,classification_orient_targets,classification_norm_targets,optimizer,total_loss)
		

		# if epoch%args.save_epoch==0:
		# 	avg_loss=avg_loss/math.ceil(len(targets)/args.batch_size_train)
		# 	avg_out_norm_loss=avg_out_norm_loss/math.ceil(len(targets)/args.batch_size_train)
		# 	avg_cross=avg_cross/math.ceil(len(targets)/args.batch_size_train)
		# 	avg_line_to_obs_loss=avg_line_to_obs_loss/math.ceil(len(targets)/args.batch_size_train)
		# 	avg_pp_to_gt_loss=avg_pp_to_gt_loss/math.ceil(len(targets)/args.batch_size_train)
		# 	# avg_point_error=avg_point_error/math.ceil(len(targets)/args.batch_size_train)
			
		# 	model_path='MLP_E2E_cross_mask_concave2d_epoch=%d_N_%d_Np_%d_avg_loss=%.3f_avg_out_norm_loss=%.3f_avg_cross=%.3f_avg_line_to_obs_loss=%.3f_avg_pp_to_gt_loss=%.3f.pkl'\
		# 		%(epoch,args.load_data_N,args.load_data_NP,avg_loss,avg_out_norm_loss,avg_cross,avg_line_to_obs_loss,\
        #            avg_pp_to_gt_loss)
			
		# 	torch.save(mlp.state_dict(),os.path.join(args.model_path,model_path))

		# 	model_path='CNN_E2E_cross_mask_concave2d_epoch=%d_N_%d_Np_%d_avg_loss=%.3f_avg_out_norm_loss=%.3f_avg_cross=%.3f_avg_line_to_obs_loss=%.3f_avg_pp_to_gt_loss=%.3f.pkl'\
		# 		%(epoch,args.load_data_N,args.load_data_NP,avg_loss,avg_out_norm_loss,avg_cross,avg_line_to_obs_loss,\
        #            avg_pp_to_gt_loss)
		# 	torch.save(encoder.state_dict(),os.path.join(args.model_path,model_path))
		# for parameters in mlp.parameters():
		# 	print("print(parameters)")
		# 	print(parameters)
		# Save the models
		if epoch % args.test_epochs ==0:
			if args.seen_environment_test:
				Tavg_error=0.0
				Tavg_out_norm_error=0.0
				Tavg_cross_error=0.0
				Tavg_line_to_obs_collision=0.0
				Tavg_pp_to_gt_error=0.0
				Tavg_Pre_point_error=0.0
				
				Ttotal_error,Tavg_error,Tavg_out_norm_error,Tavg_cross_error,Tavg_line_to_obs_collision,Tavg_pp_to_gt_error,Tavg_Pre_point_error=Test(mlp,encoder,Tenviro_mask_set,Tboundary_points_set,Tboundary_lengths,Tenv_indices,Ttargets,Tpadded_targets_future_all,Torient_dataset,Tclassification_orient_targets,Tclassification_norm_targets,Ttotal_error)
				
				print("***************")
				print ("--Taverage_error:") #change by z
				print (Tavg_error/(len(Ttargets)/args.batch_size_test))#change by z
				print ("--Taverage_out_norm_error:") #change by z
				print (Tavg_out_norm_error/(len(Ttargets)/args.batch_size_test))#change by z
				print ("--Taverage_cross_error:") #change by z
				print (Tavg_cross_error/(len(Ttargets)/args.batch_size_test))#change by z
				print ("--Taverage_line_to_obs_min_collision_error:") #change by z
				print (Tavg_line_to_obs_collision/(len(Ttargets)/args.batch_size_test))#change by z
				print ("--Taverage pp_to_gt_error:") #change by z
				print (Tavg_pp_to_gt_error/math.ceil(len(Ttargets)/args.batch_size_test))#change by z
	
				# print ("--Taverage_Pre_point_error:") #chaj2hjnge by z
				# print (Tavg_Pre_point_error/(len(Ttargets)/args.batch_size_test))#change by z
	
			if args.Unseen_environment_test:
				UTavg_error=0.0
				UTavg_out_norm_error=0.0
				UTavg_cross_error=0.0
				UTavg_line_to_obs_collision=0.0
				UTavg_Pre_point_error=0.0

				UTtotal_error,UTavg_error,UTavg_out_norm_error,UTavg_cross_error,UTavg_line_to_obs_collision,UTavg_pp_to_gt_error,UTavg_Pre_point_error=Test(mlp,encoder,UTenviro_mask_set,UTboundary_points_set,UTboundary_lengths,UTenv_indices,UTtargets,UTpadded_targets_future_all,UTorient_dataset,UTclassification_orient_targets,UTclassification_norm_targets,UTtotal_error)
				# UTtotal_error,UTavg_error,UTavg_out_norm_error,UTavg_cross_error,UTavg_line_to_obs_min_collision,UTavg_Pre_point_error=Test(mlp,obc,env_indices,targets,orient_dataset,classification_orient_targets,classification_norm_targets,Ttotal_error)
				print("***************")
				print ("--UTaverage_error:") #change by z
				print (UTavg_error/(len(UTtargets)/args.batch_size_test))#change by z
				print ("--UTaverage_out_norm_error:") #change by z
				print (UTavg_out_norm_error/(len(UTtargets)/args.batch_size_test))#change by z
				print ("--UTaverage_cross_error:") #change by z
				print (UTavg_cross_error/(len(UTtargets)/args.batch_size_test))#change by z
				print ("--UTaverage_line_to_obs_min_collision error:") #change by z
				print (UTavg_line_to_obs_collision/(len(UTtargets)/args.batch_size_test))#change by z
				print ("--UTaverage pp_to_gt_error:") #change by z
				print (UTavg_pp_to_gt_error/math.ceil(len(UTtargets)/args.batch_size_test))#change by z
				# print ("--UTaverage_Pre_point_error:") #change by z
				# print (UTavg_Pre_point_error/(len(UTtargets)/args.batch_size_test))#change by z
				
		if epoch%args.save_epoch==0 and args.seen_environment_test and args.Unseen_environment_test:
			avg_loss=avg_loss/math.ceil(len(targets)/args.batch_size_train)
			avg_out_norm_loss=avg_out_norm_loss/math.ceil(len(targets)/args.batch_size_train)
			avg_cross=avg_cross/math.ceil(len(targets)/args.batch_size_train)
			avg_line_to_obs_loss=avg_line_to_obs_loss/math.ceil(len(targets)/args.batch_size_train)
			avg_pp_to_gt_loss=avg_pp_to_gt_loss/math.ceil(len(targets)/args.batch_size_train)

			Tavg_error=Tavg_error/math.ceil(len(Ttargets)/args.batch_size_test)
			Tavg_out_norm_error=Tavg_out_norm_error/math.ceil(len(Ttargets)/args.batch_size_test)
			Tavg_cross_error=Tavg_cross_error/math.ceil(len(Ttargets)/args.batch_size_test)
			Tavg_line_to_obs_collision=Tavg_line_to_obs_collision/math.ceil(len(Ttargets)/args.batch_size_test)
			Tavg_pp_to_gt_error=Tavg_pp_to_gt_error/math.ceil(len(Ttargets)/args.batch_size_test)

			UTavg_error=UTavg_error/math.ceil(len(UTtargets)/args.batch_size_test)
			UTavg_out_norm_error=UTavg_out_norm_error/math.ceil(len(UTtargets)/args.batch_size_test)
			UTavg_cross_error=UTavg_cross_error/math.ceil(len(UTtargets)/args.batch_size_test)
			UTavg_line_to_obs_collision=UTavg_line_to_obs_collision/math.ceil(len(UTtargets)/args.batch_size_test)
			UTavg_pp_to_gt_error=UTavg_pp_to_gt_error/math.ceil(len(UTtargets)/args.batch_size_test)


			model_path='MLP_E2E_mask_2d_epoch=%d_N_%d_Np_%d_DropMLP=%.2f \
				Loss=%.2f_norm=%.2f_cross=%.2f_obs=%.2f_pptogt=%.2f \
				Te=%.2f_Tnorm=%.2f_Tcross=%.2f_Tobs=%.2f_Tpp=%.2f \
				UTe=%.2f_UTnorm=%.2f_UTcross=%.2f_UTobs=%.2f_UTpp=%.2f.pkl'\
				%(epoch,args.load_data_N,args.load_data_NP,args.dropout_p_MLP,\
      			avg_loss,avg_out_norm_loss,avg_cross,avg_line_to_obs_loss,avg_pp_to_gt_loss,\
				Tavg_error,Tavg_out_norm_error,Tavg_cross_error,Tavg_line_to_obs_collision,Tavg_pp_to_gt_error,\
				UTavg_error,UTavg_out_norm_error,UTavg_cross_error,UTavg_line_to_obs_collision,UTavg_pp_to_gt_error)
			
			torch.save(mlp.state_dict(),os.path.join(args.model_path,model_path))

			model_path='CNN_E2E_mask_2d_epoch=%d_N_%d_Np_%d_DropMLP=%.2f \
				Loss=%.2f_norm=%.2f_cross=%.2f_obs=%.2f_pptogt=%.2f \
				Te=%.2f_Tnorm=%.2f_Tcross=%.2f_Tobs=%.2f_Tpp=%.2f \
				UTe=%.2f_UTnorm=%.2f_UTcross=%.2f_UTobs=%.2f_UTpp=%.2f.pkl'\
				%(epoch,args.load_data_N,args.load_data_NP,args.dropout_p_MLP,\
      			avg_loss,avg_out_norm_loss,avg_cross,avg_line_to_obs_loss,avg_pp_to_gt_loss,\
				Tavg_error,Tavg_out_norm_error,Tavg_cross_error,Tavg_line_to_obs_collision,Tavg_pp_to_gt_error,\
				UTavg_error,UTavg_out_norm_error,UTavg_cross_error,UTavg_line_to_obs_collision,UTavg_pp_to_gt_error)
			
			torch.save(encoder.state_dict(),os.path.join(args.model_path,model_path))
			
		

	# torch.save(total_loss,'new_total_loss.dat')
	# model_path='new_mlp_cross_mask_concave2d_100_4000_PReLU_ae_dd_final.pkl'
	# torch.save(mlp.state_dict(),os.path.join(args.model_path,model_path))
if __name__ == '__main__':
	parser = argparse.ArgumentParser()	
	parser.add_argument('--model_path', type=str, default='./models/crossentropy/test',help='path for saving trained models')
	# Model parameters
	parser.add_argument('--input_size', type=int , default=32, help='dimension of the input vector')
	# Mask map parameters
	parser.add_argument('--scale_para', type=float , default=4, help='the scale of mask map')
	parser.add_argument('--bias', type=float , default=20.0, help='the bias of mask map')


#____________________________
	# the parameters under this line need to be adjust
	parser.add_argument('--num_epochs', type=int, default=4000)
	parser.add_argument('--test_epochs', type=int, default=10,help="Perform testing every test_epochs epochs.") 
	parser.add_argument('--seen_environment_test', type=int, default=0,help='Use the seen environment to test')
	parser.add_argument('--Unseen_environment_test', type=int, default=0,help='Use the Unseen environment to test')
	parser.add_argument('--preload_all', type=int, default=0,help='preload the encoderand mlp parameter or not')
	parser.add_argument('--preload_encoder', type=int, default=0,help='preload the encoder parameter or not')
	parser.add_argument('--learning_rate_together', type=float, default=0.0001)  #0.0001 before concave
	parser.add_argument('--learning_rate_encoder', type=float, default=0.00001)  #0.0001 before concave
	parser.add_argument('--learning_rate_planner', type=float, default=0.0001)  #0.0001 before concave
	
	parser.add_argument('--num_sector', type=int , default=180, help='The number of sectors that divide the space and num_sector must be even number') 
	parser.add_argument('--batch_size_train', type=int, default=100)
	parser.add_argument('--batch_size_test', type=int, default=100,help='test during the training')
	parser.add_argument('--interpolation_point_num', type=int, default=50,help='interpolation_point_num in loss function')
	parser.add_argument('--dropout_p_MLP', type=float , default=0.0, help='The probability of dropout')
	parser.add_argument('--dropout_p_CNN', type=float , default=0.0, help='The probability of dropout')
	# parser.add_argument('--mask_size', type=int , default=160, help='The size of mask map')
	
	parser.add_argument('--save_epoch', type=int, default=50, help='The frequency of saving the model')
	
	
	#data size for load and test
	parser.add_argument('--load_data_N', type=int, default=10)
	parser.add_argument('--load_data_NP', type=int, default=4)

	parser.add_argument('--T_data_N', type=int, default=10,help='test seen environment')
	parser.add_argument('--T_data_NP', type=int, default=40,help='test seen environment')
	parser.add_argument('--T_data_s', type=int, default=0,help='test seen environment')
	parser.add_argument('--T_data_sp', type=int, default=0,help='test seen environment')

	parser.add_argument('--UT_data_N', type=int, default=10,help='test Unseen environment')
	parser.add_argument('--UT_data_NP', type=int, default=40,help='test Unseens environment')
	parser.add_argument('--UT_data_s', type=int, default=0,help='test Unseen environment')
	parser.add_argument('--UT_data_sp', type=int, default=0,help='test Unseen environment')

	# parser.add_argument('--T_data_N', type=int, default=10,help='test seen environment')
	# parser.add_argument('--T_data_NP', type=int, default=5,help='test seen environment')
	# parser.add_argument('--T_data_s', type=int, default=90,help='test seen environment')
	# parser.add_argument('--T_data_sp', type=int, default=4000,help='test seen environment')

	# parser.add_argument('--UT_data_N', type=int, default=10,help='test Unseen environment')
	# parser.add_argument('--UT_data_NP', type=int, default=5,help='test Unseens environment')
	# parser.add_argument('--UT_data_s', type=int, default=100,help='test Unseen environment')
	# parser.add_argument('--UT_data_sp', type=int, default=0,help='test Unseen environment')


	#parameters for the loss_function
	parser.add_argument('--k1', type=float, default=1,help='parameter of out_norm_loss')
	parser.add_argument('--k2', type=float, default=4.0,help='parameter of loss_cross')
	parser.add_argument('--k3', type=float, default=0.1,help='parameter of line_to_obs_min_collision')
	parser.add_argument('--k4', type=float, default=2,help='parameter of predict_point_to_gt')
	#parameters for the error_function  just influence the number of parameter error_all in test presentation
	parser.add_argument('--Tk1', type=float, default=1.0,help='parameter of out_norm_mse')
	parser.add_argument('--Tk2', type=float, default=4.0,help='parameter of loss_cross')
	parser.add_argument('--Tk3', type=float, default=0.1,help='parameter of line_to_obs_min_collision')
	parser.add_argument('--Tk4', type=float, default=2,help='parameter of predict_point_to_gt')

	parser.add_argument('--ek', type=float, default=2,help='parameter of loss_function')
	
# #____________________________
# 	# the parameters under this line need to be adjust
# 	parser.add_argument('--num_epochs', type=int, default=5000)
# 	parser.add_argument('--test_epochs', type=int, default=10,help="Perform testing every test_epochs epochs.") 
# 	parser.add_argument('--preload', type=bool, default=True,help='preload the mlp parameter or not')
# 	parser.add_argument('--seen_environment_test', type=bool, default=True,help='Use the seen environment to test')
# 	parser.add_argument('--Unseen_environment_test', type=bool, default=True,help='Use the Unseen environment to test')

# 	parser.add_argument('--num_sector', type=int , default=180, help='The number of sectors that divide the space and num_sector must be even number') 
# 	parser.add_argument('--batch_size_train', type=int, default=100)
# 	parser.add_argument('--batch_size_test', type=int, default=100,help='test during the training')
# 	parser.add_argument('--learning_rate', type=float, default=0.0001)  #0.0001 before concave
# 	parser.add_argument('--interpolation_point_num', type=int, default=50,help='interpolation_point_num in loss function')
# 	parser.add_argument('--dropout_p', type=float , default=0.0, help='The probability of dropout')

	
	
# 	#data size for load and test
# 	parser.add_argument('--load_data_N', type=int, default=100)
# 	parser.add_argument('--load_data_NP', type=int, default=400)

# 	parser.add_argument('--T_data_N', type=int, default=100,help='test seen environment')
# 	parser.add_argument('--T_data_NP', type=int, default=10,help='test seen environment')
# 	parser.add_argument('--T_data_s', type=int, default=0,help='test seen environment')
# 	parser.add_argument('--T_data_sp', type=int, default=4000,help='test seen environment')

# 	parser.add_argument('--UT_data_N', type=int, default=10,help='test Unseen environment')
# 	parser.add_argument('--UT_data_NP', type=int, default=100,help='test Unseens environment')
# 	parser.add_argument('--UT_data_s', type=int, default=100,help='test Unseen environment')
# 	parser.add_argument('--UT_data_sp', type=int, default=0,help='test Unseen environment')


# 	#parameters for the loss_function
# 	parser.add_argument('--k1', type=float, default=1.0,help='parameter of out_norm_mse')
# 	parser.add_argument('--k2', type=float, default=1.0,help='parameter of loss_cross')
# 	parser.add_argument('--k3', type=float, default=0.1,help='parameter of line_to_obs_min_collision')
# 	#parameters for the error_function  just influence the number of parameter error_all in test presentation
# 	parser.add_argument('--Tk1', type=float, default=1.0,help='parameter of out_norm_mse')
# 	parser.add_argument('--Tk2', type=float, default=1.0,help='parameter of loss_cross')
# 	parser.add_argument('--Tk3', type=float, default=0.1,help='parameter of line_to_obs_min_collision')

# 	parser.add_argument('--ek', type=float, default=2,help='parameter of loss_function')
	
# #------------------------------------------------------------------------------------------------

	# parser.add_argument('--no_env', type=int, default=50,help='directory for obstacle images')
	# parser.add_argument('--no_motion_paths', type=int,default=2000,help='number of optimal paths in each environment')
	# parser.add_argument('--log_step', type=int , default=10,help='step size for prining log info')
	# parser.add_argument('--save_step', type=int , default=1000,help='step size for saving trained models')
	# parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
	# parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
	# parser.add_argument('--num_layers', type=int , default=4, help='number of layers in lstm')
	args = parser.parse_args()
	print(args)
	main(args)

# it is for orientation version

