import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_test_dataset_planner 
from model import MLP_3D 
import AE_R_3d_CNN.CNN_3d as AE
from model import MLP_original
from torch.autograd import Variable 
import math
import time

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

scale_para=3.2
bias=20.0
enviro_type="Unseen"
#enviro_type="Seen"
#load test dataset
if enviro_type=="Unseen":
	#Uneen
	 enviro_mask_set,obc, paths, path_lengths,obs_start= load_test_dataset_planner(scale_para=scale_para,bias=bias,N=5, NP=1000,s=30, sp=1800) 
	# enviro_mask_set,obc, paths, path_lengths,obs_start,obs_recover= load_test_dataset_planner(scale_para=scale_para,bias=bias,N=10, NP=1000,s=100, sp=0) 
else:
	# #Seen
	# enviro_mask_set,obc,obstacles, paths, path_lengths,obs_start,obs_recover= load_test_dataset_planner(scale_para=3.5,bias=20.0,N=100,NP=200, s=0,sp=4000) 
	enviro_mask_set,obc, paths, path_lengths,obs_start= load_test_dataset_planner(scale_para=scale_para,bias=bias,N=30,NP=1000, s=0,sp=100) 

def angle_norm_to_coordinate(out_norm,out_angle_theta,out_angle_phi,bi):
	
	theta = out_angle_theta / 360 * 2 * math.pi
	phi = out_angle_phi / 360 * 2 * math.pi

	x = bi[60] + torch.sin(theta) * torch.cos(phi) * out_norm
	y = bi[61]+ torch.sin(theta) * torch.sin(phi) * out_norm
	z = bi[62] + torch.cos(theta) * out_norm	

	predict_point = torch.cat((x, y, z)) 
		
	return predict_point

def orient_norm_to_coordinate(orient_norm,last_point):
	next_point=torch.tensor([0.0,0.0])
	next_point[0]=last_point[0]+math.cos(orient_norm[0]/360*2*math.pi)*orient_norm[1]
	next_point[1]=last_point[1]+math.sin(orient_norm[0]/360*2*math.pi)*orient_norm[1]
	return next_point

def r3d_plot_path(fp,path,size_start_end,marker_start,color,plot_color_count,marker_end,size_generate,paths,path_lengths,size_act,i,j,ax1):
	print(path)

	if fp==1: #feasible path without replan
		ax1.scatter(path[0,0], path[0,1],path[0,2], s=size_start_end,marker=marker_start, c=color[plot_color_count])
		ax1.scatter(path[len(path)-1,0], path[len(path)-1,1],path[len(path)-1,2], s=size_start_end,marker=marker_end,c=color[plot_color_count])
		ax1.scatter(path[:,0], path[:,1],path[:,2],s=size_generate,marker='o', c=color[plot_color_count])
		ax1.plot(path[:,0], path[:,1],path[:,2],linewidth=3.0,c=color[plot_color_count])
		#demonstrate the BIT* paths
		ax1.scatter(paths[i][j][:path_lengths[i][j]][:,0], paths[i][j][:path_lengths[i][j]][:,1], paths[i][j][:path_lengths[i][j]][:,2],s=size_act,marker='d', c=color[plot_color_count]) #!!!!!!!
		ax1.plot(paths[i][j][:path_lengths[i][j]][:,0], paths[i][j][:path_lengths[i][j]][:,1], paths[i][j][:path_lengths[i][j]][:,2],linestyle=':',c=color[plot_color_count])
	elif fp==0: #can not get a feasible path
		ax1.scatter(path[0,0], path[0,1],path[0,2], s=size_start_end,marker=marker_start, c=color[plot_color_count])
		ax1.scatter(path[len(path)-1,0], path[len(path)-1,1],path[len(path)-1,2], s=size_start_end,marker=marker_end,c=color[plot_color_count])
		ax1.scatter(path[:,0], path[:,1],path[:,2],s=size_generate,marker='x', c=color[plot_color_count])
		ax1.plot(path[:,0], path[:,1],path[:,2],linewidth=3.0,c=color[plot_color_count])
		#demonstrate the BIT* paths
		ax1.scatter(paths[i][j][:path_lengths[i][j]][:,0], paths[i][j][:path_lengths[i][j]][:,1], paths[i][j][:path_lengths[i][j]][:,2],s=size_act,marker='d', c=color[plot_color_count]) #!!!!!!!
		ax1.plot(paths[i][j][:path_lengths[i][j]][:,0], paths[i][j][:path_lengths[i][j]][:,1], paths[i][j][:path_lengths[i][j]][:,2],linestyle=':',c=color[plot_color_count])
	elif fp==2: #replan
		ax1.scatter(path[0,0], path[0,1],path[0,2], s=size_start_end,marker=marker_start, c=color[plot_color_count])
		ax1.scatter(path[len(path)-1,0], path[len(path)-1,1],path[len(path)-1,2], s=size_start_end,marker=marker_end,c=color[plot_color_count])
		ax1.scatter(path[:,0], path[:,1],path[:,2],s=size_generate,marker='*', c=color[plot_color_count])
		ax1.plot(path[:,0], path[:,1],path[:,2],linewidth=3.0,c=color[plot_color_count])
		#demonstrate the BIT* paths
		ax1.scatter(paths[i][j][:path_lengths[i][j]][:,0], paths[i][j][:path_lengths[i][j]][:,1], paths[i][j][:path_lengths[i][j]][:,2],s=size_act,marker='d', c=color[plot_color_count]) #!!!!!!!
		ax1.plot(paths[i][j][:path_lengths[i][j]][:,0], paths[i][j][:path_lengths[i][j]][:,1], paths[i][j][:path_lengths[i][j]][:,2],linestyle=':',c=color[plot_color_count])

	return 1

def r3d_plot_cloud(obs,i,s,ax1):
    # could_plt=obs[i].detach().cpu().numpy()
    temp=np.fromfile('../../DATA_SET/r3d/obs_cloud_3D_narrow/obc_3D_narrow'+str(i+s)+'.dat') 
    could_plt=temp.reshape(len(temp)//3,3) 		
    ax1.scatter(could_plt[:,0], could_plt[:,1],could_plt[:,2], c='b')
    ax1.set_xlabel('X label') 
    ax1.set_ylabel('Y label')
    ax1.set_zlabel('Z label')

    # print("obs=")
    # print(could_plt)
    return 1

def r3d_count_length(path,paths,i,j,path_lengths):
    path_cost=0
    expert_path_cost=0
    expert_path=paths[i][j][:path_lengths[i][j]]
    for k in range (0,len(path)-1):
        path_cost=path_cost+((path[k+1][0]-path[k][0])**2+(path[k+1][1]-path[k][1])**2+(path[k+1][2]-path[k][2])**2)**0.5
        
    for m in range (0,path_lengths[i][j]-1):
        expert_path_cost=expert_path_cost+((expert_path[m+1][0]-expert_path[m][0])**2+(expert_path[m+1][1]-expert_path[m][1])**2+(expert_path[m+1][2]-expert_path[m][2])**2)**0.5
        # paths[i][j][:path_lengths[i][j]][:,0]
    rate=path_cost/expert_path_cost
    # print("expert_path=====")
    # print(expert_path)
    # print("expert_path_cost=====")
    # print("path_cost=====")
    # print(path_cost)
    # print("expert_path_cost=====")
    # print(expert_path_cost)
    # print("rate======")
    # print(rate)

    return path_cost,expert_path_cost,rate

#arbitrary obstacle type
def IsInCollision(x,idx): 
	
	cf=False
	x=(x+bias)*scale_para
	 
	if isinstance(x, torch.Tensor):
		x=x.detach().numpy()

	fx=np.floor(x).astype(int)
	cx=np.ceil(x).astype(int)
	if fx[0]>=scale_para*40:
		fx[0]=scale_para*40-1
	if fx[1]>=scale_para*40:
		fx[1]=scale_para*40-1
	if fx[2]>=scale_para*40:
		fx[2]=scale_para*40-1
	if cx[0]>=scale_para*40:
		cx[0]=scale_para*40-1
	if cx[1]>=scale_para*40:
		cx[1]=scale_para*40-1
	if cx[2]>=scale_para*40:
		cx[2]=scale_para*40-1
	if enviro_mask_set[idx,fx[0],fx[1],fx[2]] and enviro_mask_set[idx,cx[0],cx[1],cx[2]]:

		return True

	return False




def steerTo (start, end, idx):

	DISCRETIZATION_STEP=0.01
	dists=np.zeros(3,dtype=np.float32)
	for i in range(0,3): 
		dists[i] = end[i] - start[i]

	distTotal = 0.0
	for i in range(0,3): 
		distTotal =distTotal+ dists[i]*dists[i]


	distTotal = math.sqrt(distTotal)
	if distTotal>0:
		incrementTotal = distTotal/DISCRETIZATION_STEP
		for i in range(0,3): 
			dists[i] =dists[i]/incrementTotal


		numSegments = int(math.floor(incrementTotal))

		stateCurr = np.zeros(3,dtype=np.float32)
		for i in range(0,3): 
			stateCurr[i] = start[i]
		for i in range(0,numSegments):

			if IsInCollision(stateCurr,idx):
				return 0

			for j in range(0,3):
				stateCurr[j] = stateCurr[j]+dists[j]


		if IsInCollision(end,idx):
			return 0


	return 1

# checks the feasibility of entire path including the path edges
def feasibility_check(path,idx):

	for i in range(0,len(path)-1):
		ind=steerTo(path[i],path[i+1],idx)
		if ind==0:
			return 0
	return 1


# checks the feasibility of path nodes only
def collision_check(path,idx):

	for i in range(0,len(path)):
		if IsInCollision(path[i],idx):
			return 0
	return 1

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def get_input(i,dataset,targets,seq,bs):
	bi=np.zeros((bs,18),dtype=np.float32)
	bt=np.zeros((bs,2),dtype=np.float32)
	k=0	
	for b in range(i,i+bs):
		bi[k]=dataset[seq[i]].flatten()
		bt[k]=targets[seq[i]].flatten()
		k=k+1
	return torch.from_numpy(bi),torch.from_numpy(bt)



def is_reaching_target(start1,start2):
	s1=np.zeros(2,dtype=np.float32)
	s1[0]=start1[0]
	s1[1]=start1[1]

	s2=np.zeros(2,dtype=np.float32)
	s2[0]=start2[0]
	s2[1]=start2[1]


	for i in range(0,2):
		if abs(s1[i]-s2[i]) > 1.0: 
			return False
	return True

#lazy vertex contraction 
def lvc(path,idx):

	for i in range(0,len(path)-1):
		for j in range(len(path)-1,i+1,-1):
			ind=0
			ind=steerTo(path[i],path[j],idx)
			if ind==1:
				pc=[]
				for k in range(0,i+1):
					pc.append(path[k])
				for k in range(j,len(path)):
					pc.append(path[k])

				return lvc(pc,idx)
				
	return path


def replan_path(mlp,p,g,idx,obs):
	step=0
	iteration_replan=0
	path=[]
	path.append(p[0])
	for i in range(1,len(p)-1):
		if not IsInCollision(p[i],idx):   #cut the collision point
			path.append(p[i])
	path.append(g)			
	new_path=[]
	for i in range(0,len(path)-1):
		target_reached=False

	 
		st=path[i]
		gl=path[i+1]
		steer=steerTo(st, gl, idx)
		if steer==1:
			new_path.append(st)
			new_path.append(gl)
		else:
			# print("replanning work and path[i] and path[i+1] is:")
			# print(path[i])
			# print(path[i+1])
			itr=0
			pA=[]
			pA.append(st)
			pB=[]
			pB.append(gl)
			target_reached=0
			tree=0
			rep_stuck_flag_forward=1 #it has been stuck for one time in the original mlp
			rep_stuck_flag_backward=1 #it has been stuck for one time in the original mlp
			bias_angle_backward=0
			stop_forward=0
			stop_backward=0
			out_range_forward=0
			out_range_backward=0

			while target_reached==0 and itr<2000 :
			#while target_reached==0:
				if stop_forward==1 and stop_backward==1:
					break
				itr=itr+1
				print("itr=")
				print(itr)
				if tree==0:
					
					if stop_forward==0:
						st_pre=st
						ip1=torch.cat((obs,st,gl))
						ip1=to_var(ip1)

						st,out_range_forward=Re_Input_to_output_point(mlp,ip1,args.num_sector_theta,args.num_sector_phi,rep_stuck_flag_forward,out_range_forward)
						iteration_replan+=1
						# st=st.data.cpu()
						#pA.append(st)
						#steerTo(st_pre, st, idx) the St_pre needs to be before st in order to match the function featability_check
						if IsInCollision(st,idx) or not steerTo(st_pre, st, idx) or out_range_forward>2:
								st=st_pre
								rep_stuck_flag_forward=rep_stuck_flag_forward+1
								if rep_stuck_flag_forward>args.stuck_search_time-1:
									rep_stuck_flag_forward=args.stuck_search_time-1
									print("rep_stuck_flag_forward>args.stuck_search_time!!!!!!!!!")
									stop_forward=1
									
								
						else:
								pA.append(st)
								rep_stuck_flag_forward=0
								stop_forward=0
								# print("fffff")
								# print(pA)
					print("rep_stuck_flag_forward")
					print(rep_stuck_flag_forward)
					print("out_range_forward")
					print(out_range_forward)
					tree=1
				else:			
					if stop_backward==0:
						gl_pre=gl
						ip2=torch.cat((obs,gl,st))
						ip2=to_var(ip2)

						gl,out_range_backward=Re_Input_to_output_point(mlp,ip2,args.num_sector_theta,args.num_sector_phi,rep_stuck_flag_backward,out_range_backward)
						iteration_replan+=1
						# gl=mlp(ip2)
						# gl=gl.data.cpu()
						print("gl_forward")
						print(gl)

						#pB.append(gl)
						##steerTo(gl, gl_pre, idx) the gl needs to be before gl_pre in order to match the function featability_check 
						if IsInCollision(gl,idx) or not steerTo(gl, gl_pre, idx) or out_range_backward>2:
								gl=gl_pre
								rep_stuck_flag_backward=rep_stuck_flag_backward+1
								if rep_stuck_flag_backward>args.stuck_search_time-1:
									rep_stuck_flag_backward=args.stuck_search_time-1
									print("rep_stuck_flag_backward>args.stuck_search_time!!!!!!!!!")
									stop_backward=1
						else:
								pB.append(gl)
								rep_stuck_flag_backward=0
								stop_backward=0
								# print("bbbbbbb")
								# print(pB)
						
					print("rep_stuck_flag_backward")
					print(rep_stuck_flag_backward)
					print("out_range_backward")
					print(out_range_backward)
					tree=0		
				target_reached=steerTo(st, gl, idx)
			print("itr=")
			print(itr)
			if target_reached==0:
				
				print("Replanning module discover two disconnected points") #change by z
				
				#return 0
			#else:
			if 1:
				# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
				#Although some elements will be repeatedly pushed into new_path, 
				# it will only have a slight time impact on the final result
				for p1 in range(0,len(pA)):
					new_path.append(pA[p1])
				for p2 in range(len(pB)-1,-1,-1):
					new_path.append(pB[p2])
			
	return new_path,iteration_replan
    

def max_out_softm_to_angle(num_sector_theta,num_sector_phi,out_orient_theta_softm, out_orient_phi_softm):
	
	max_possibility_theta, max_index_theta = torch.max(out_orient_theta_softm, dim=-1)
	max_possibility_phi, max_index_phi = torch.max(out_orient_phi_softm, dim=-1)


	center_angle_theta = np.linspace(0+180/num_sector_theta/2, 180-180/num_sector_theta/2, num_sector_theta) #center_angle for each sector
	center_angle_theta = torch.from_numpy(center_angle_theta).float().cuda()
	
	center_angle_phi = np.linspace(-180+360/num_sector_phi/2, 180-360/num_sector_phi/2, num_sector_phi) #center_angle for each sector
	center_angle_phi = torch.from_numpy(center_angle_phi).float().cuda()

    # A model that take the most probable angle #for multi-dimension out_orient_softm
	#out_angle_max=np.zeros((len(out_orient_softm)),dtype=np.float32)
	#out_angle_max = torch.from_numpy(out_angle_max).float().cuda()
	# for i in range(0,len(out_orient_softm)):
	# 	out_angle_max[i]=center_angle[max_index[i]]
	# out_angle_max=out_angle_max.unsqueeze(1)
 
	out_angle_theta_max=center_angle_theta[max_index_theta] #for 1 dimension
	out_angle_phi_max=center_angle_phi[max_index_phi] #for 1 dimension

	# # A model that considers all angle possibilities
	# center_angle = center_angle.unsqueeze(0) #cuda
	# out_angle = torch.mul(center_angle, out_orient_softm) # (batch_size,num_sector)
	# out_angle = torch.sum(out_angle, dim=-1) # (batch_size,)
	# out_angle=out_angle.unsqueeze(1) #add a dimension
	
	return out_angle_theta_max,out_angle_phi_max


def rep_out_softm_to_angle(num_sector_theta,num_sector_phi,out_orient_theta_softm, out_orient_phi_softm,rep_stuck_flag):
		
	N=1+rep_stuck_flag*args.Sparse_index #Find the element ranked rep_suck_flag+1
	# start_time = time.time()
	prob_matrix = out_orient_theta_softm[:, None] * out_orient_phi_softm[None, :]
	values_all, indices_all = torch.topk(prob_matrix.view(-1), N) #the first one is 0
	# nth_value = values[N-1]  
	nth_index = indices_all[N-1] #Because the index starts from 0
	nth_index_theta, nth_index_phi = nth_index // out_orient_phi_softm.size(0), nth_index % out_orient_phi_softm.size(0)
	# end_time = time.time()

	# value_theta, indices_theta = torch.topk(out_orient_theta_softm,rep_stuck_flag+1)
	# value_phi, indices_phi = torch.topk(out_orient_phi_softm,rep_stuck_flag+1)

	center_angle_theta = np.linspace(0+180/num_sector_theta/2, 180-180/num_sector_theta/2, num_sector_theta) #[0,pi]
	center_angle_theta = torch.from_numpy(center_angle_theta).float().cuda()
	
	center_angle_phi = np.linspace(-180+360/num_sector_phi/2, 180-360/num_sector_phi/2, num_sector_phi) #center_angle for each sector #[-pi,pi]
	center_angle_phi = torch.from_numpy(center_angle_phi).float().cuda()
	
	out_angle_theta_rep=center_angle_theta[nth_index_theta] #for 1 dimension
	out_angle_phi_rep=center_angle_phi[nth_index_phi] #for 1 dimension
	
		
	return out_angle_theta_rep,out_angle_phi_rep

def Range_Limit(point,out_range):
	count_flag=0
	limit=20
	for i in range (0,3):
		if point[i]>limit:
			point[i]=limit	
			count_flag=1
			print("!!!!!!!!!!!!!!!!!!!!!!!!!out of range!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		if point[i]<-limit:
			point[i]=-limit
			count_flag=1
			print("!!!!!!!!!!!!!!!!!!!!!!!!!out of range!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	if count_flag==1:
		out_range=out_range+1
	else:
		out_range=0
	return point,out_range

def Input_to_output_point(mlp,input,num_sector_theta,num_sector_phi,out_range):
	out_norm, out_orient_theta_raw,out_orient_phi_raw,\
			  out_orient_theta_softm, out_orient_phi_softm=mlp(input)
	out_angle_theta_max,out_angle_phi_max=max_out_softm_to_angle(num_sector_theta,num_sector_phi,out_orient_theta_softm, out_orient_phi_softm)

	out_norm=out_norm.data.cpu()
	out_angle_theta_max=out_angle_theta_max.data.cpu()
	out_angle_phi_max=out_angle_phi_max.data.cpu()
	output_point=angle_norm_to_coordinate(out_norm,out_angle_theta_max,out_angle_phi_max,input)
	output_point,out_range=Range_Limit(output_point,out_range)
	return output_point,out_range

def Re_Input_to_output_point(mlp,input,num_sector_theta,num_sector_phi,rep_stuck_flag_forward,out_range):
	out_norm, out_orient_theta_raw,out_orient_phi_raw,\
			  out_orient_theta_softm, out_orient_phi_softm=mlp(input)
	out_angle_theta_rep,out_angle_phi_rep=rep_out_softm_to_angle(num_sector_theta,num_sector_phi,out_orient_theta_softm, out_orient_phi_softm,\
				  rep_stuck_flag_forward)
	out_norm=0.51*(out_norm)+0.49*(out_norm)*math.sin(((rep_stuck_flag_forward/args.stuck_search_time)*180+90)/360*2*math.pi)
	out_norm=out_norm.data.cpu()
	out_angle_theta_rep=out_angle_theta_rep.data.cpu()
	out_angle_phi_rep=out_angle_phi_rep.data.cpu()
	# gl_orient_norm2[1]=0.51*(gl_orient_norm2[1])+0.49*(gl_orient_norm2[1])*math.sin((bias_angle_backward+90)/360*2*math.pi)
	output_point=angle_norm_to_coordinate(out_norm,out_angle_theta_rep,out_angle_phi_rep,input)
	output_point,out_range=Range_Limit(output_point,out_range)
	return output_point,out_range

def main(args):
	# Load trained model for path generation
	mlp = MLP_3D(args.input_size,args.num_sector_theta,args.num_sector_phi,args.dropout_p)
	encoder=AE.Encoder_CNN_3D(int(scale_para*40),args.dropout_p)

	mlp.eval()
	encoder.eval()

	mlp.load_state_dict(torch.load(args.MLP_path))
	encoder.load_state_dict(torch.load(args.CNN_path))


	if torch.cuda.is_available():
		mlp.cuda()
		encoder.cuda()
	
	
	with torch.no_grad():
		# Create model directory
		if not os.path.exists(args.model_path):
			os.makedirs(args.model_path)
		
		size_generate=30
		size_act=5
		size_start_end=50
		marker_start='s'
		marker_end='v'
		color=["cyan","green","black","magenta","red","orange","purple","beige"]
			
		tp=0
		fp=0
		tot=[]
		iteration_all=[]
		path_cost_all=[]
		expert_cost_all=[]
		path_count=0
		path_feasible_count=0
		infeasible_path_index1=[]
		infeasible_path_index2=[]
		invalid_start_or_goal=[]
		each_env_count=np.zeros(args.enviro_index_up-args.enviro_index_low)     #be used for count the feasible path in each environment
		each_env_sum_rate=np.zeros(args.enviro_index_up-args.enviro_index_low)
		#for i in range(7,8): #i belongs to (0,100) and (0,10) for seen environment and unseen environment, respectively.
		for i in range(args.enviro_index_low,args.enviro_index_up): #//change by z  SEEN
		#for i in range(7,8): #//change by z  SEEN
		#if 0:
			#plot_color_count=0
			# test=torch.tensor([19.9783,  2.1122])
			# print("IsInCollision")
			# print(IsInCollision(test,i))
			# if not IsInCollision(test,i):
			# 	print("*")
			
			if args.Plot_flag:
				fig=plt.figure()
				ax1= Axes3D(fig)
				ax1.set_title("Environment %d" %i, y=1.0)  
			
			color_flag=0 #it is used to present the path points' color
			env_mask_set_i=torch.from_numpy(enviro_mask_set[i]).float().unsqueeze(0).unsqueeze(1)
			env_mask_set_i=to_var(env_mask_set_i) #[1,1,scare*40,scare*40,scare*40]
			
			CNN_mask_i=encoder(env_mask_set_i).squeeze(0).cpu()	
			print("CNN_mask_i",CNN_mask_i.shape) #[60]
			
			path_cost_env=[]
			expert_cost_env=[]
			et=[]
			iteration_env=[]
			
			#for j in list(range(51,52)) + list(range(522,524))+ list(range(570,571))+ list(range(671,672))+ list(range(721,722)):   #due to the color_flag setting, it's range would better be set no more than 6.  j belongs to (0,200) and (0,2000) for seen environment and unseen environment, respectively.		
			#for j in range(100,101): #change by z 
			#for j in range(671,672): #change by z
			#for j in list(range(103,104))+ list(range(298,299)):
			#for j in list(range(0,1))+list(range(57,58))+list(range(135,136)): #change by z
			#for j in list(range(43,45))+list(range(142,143))+list(range(234,235))+list(range(267,268))+list(range(382,383)): #change by z
			#for j in list(range(66,68)): #change by z
			#for j in range(10,17): #change by z  #change args.Plot_flag
			for j in range(args.Path_index_low,args.Path_index_up): #change by z  #change args.Plot_flag
			#for j in range(0,200): #seen
				iteration=0
				print ("path: i="+str(i)+" j="+str(j))
				print("expert_path",paths[i][j][:path_lengths[i][j]])
				#print(path_lengths[i][j])
				if path_lengths[i][j]>0:								
					start=np.zeros(3,dtype=np.float32)
					goal=np.zeros(3,dtype=np.float32)
					for l in range(0,3):
						start[l]=paths[i][j][0][l]
					
					for l in range(0,3):
						goal[l]=paths[i][j][path_lengths[i][j]-1][l]
					if IsInCollision(start,i) or IsInCollision(goal,i):
						invalid_start_or_goal.append([i,j])
						continue
					tp=tp+1
					path_count=path_count+1

					#change by z
					print("start_point"+str(start))
					print("end_point"+str(goal))
					
					#start and goal for bidirectional generation
					## starting point
					start1=torch.from_numpy(start)
					goal2=torch.from_numpy(start)
					##goal point
					goal1=torch.from_numpy(goal)
					start2=torch.from_numpy(goal)
					##obstacles
					# obs=obstacles[i]
					# obs=torch.from_numpy(obs)
					obs=CNN_mask_i
				
					
					##generated paths
					path1=[] 
					path1.append(start1)
					
					path2=[]
					path2.append(start2)
					path=[]
					target_reached=0
					step=0

					path=[] # stores end2end path by concatenating path1 and path2
					tree=0	
					tic = time.clock()	
					out_range_forward_basic=0
					out_range_backward_basic=0
					stop_forward_basic=0
					stop_backward_basic=0
					while target_reached==0 and step<50 :
						if stop_forward_basic==1 and stop_backward_basic==1:
							break
						step=step+1
						if tree==0:
							start_pre1=start1
							inp1=torch.cat((obs,start1,start2))
							inp1=to_var(inp1) #[66]
														
							start1,out_range_forward_basic=Input_to_output_point(mlp,inp1,args.num_sector_theta,args.num_sector_phi,out_range_forward_basic)
							iteration+=1
							if out_range_forward_basic>2: #50
								stop_forward_basic=1
							else:
								stop_forward_basic=0			
		
							path1.append(start1)
								
							tree=1
						else:
							start_pre2=start2
							inp2=torch.cat((obs,start2,start1))
							inp2=to_var(inp2)
							start2,out_range_backward_basic=Input_to_output_point(mlp,inp2,args.num_sector_theta,args.num_sector_phi,out_range_backward_basic)
							iteration+=1
							if out_range_backward_basic>2: #50
								stop_backward_basic=1
							else:
								stop_backward_basic=0

							path2.append(start2)
							
							tree=0
						target_reached=steerTo(start1,start2,i);				
					# print("path1")
					# print(path1)
					# print("path2")
					# print(path2)
					# print("step_ite"+"j")
					# print(step)
						
					# if 1:
					if target_reached==0:
						print("target_reached_false!")
						infeasible_path_index1.append([i,j])
						for p1 in range(0,len(path1)):
								path.append(path1[p1])
						for p2 in range(len(path2)-1,-1,-1):
								path.append(path2[p2])
						# # path=lvc(path,i)	
						# path_plt_m=torch.stack(path) 
						# path_plt=path_plt_m.detach().cpu().numpy()
						# if args.Plot_flag:
						# 	r3d_plot_path(0,path_plt,size_start_end,marker_start,color,color_flag,marker_end,size_generate,paths,path_lengths,size_act,i,j,ax1)
						# 	color_flag=color_flag+1 

					# if target_reached==1:
					if 1:
						# print("target_reached")

						for p1 in range(0,len(path1)):
							path.append(path1[p1])
						for p2 in range(len(path2)-1,-1,-1):
							path.append(path2[p2])

						if args.Path_lvc_state:
							path=lvc(path,i)  #open
						
						indicator=feasibility_check(path,i)
						
						if indicator==1:
							path_feasible_count=path_feasible_count+1
							toc = time.clock()
							t=toc-tic
							et.append(t)
							fp=fp+1				

							#change by z
							
							path_plt_m=torch.stack(path)
								
							path_plt=path_plt_m.detach().cpu().numpy()
							# print("path_test")
							# print(path_plt)
							if args.Plot_flag:
								r3d_plot_path(indicator,path_plt,size_start_end,marker_start,color,color_flag,marker_end,size_generate,paths,path_lengths,size_act,i,j,ax1)
								color_flag=color_flag+1 				

						else:
							sp=0
							indicator=0
							replan_num=1#4

							#direction=[[1,-1],[-1,1],[1,1],[-1,-1]]
						
							while indicator==0 and sp<replan_num and path !=0:  #20 is better
								#dir=direction[sp]
								print("replan_path")
								sp=sp+1
								g=np.zeros(2,dtype=np.float32)
								g=torch.from_numpy(paths[i][j][path_lengths[i][j]-1])
								# print("path that sent to replan")
								# print(path)
								path,iteration_replan=replan_path(mlp,path,g,i,obs) #replanning at coarse level
								iteration=iteration+iteration_replan
								if path !=0:
									if args.Path_lvc_state:
										path=lvc(path,i) #open
									indicator=feasibility_check(path,i)

								#change by z 
								if indicator==0:
									infeasible_path_index2.append([i,j])
									path_plt_m=torch.stack(path)	
									path_plt=path_plt_m.detach().cpu().numpy()
									if args.Plot_flag:
										r3d_plot_path(indicator,path_plt,size_start_end,marker_start,color,color_flag,marker_end,size_generate,paths,path_lengths,size_act,i,j,ax1)
										color_flag=color_flag+1 
									
								if indicator==1:
									toc = time.clock()
									t=toc-tic
									et.append(t)
									fp=fp+1
			
									#change by z
									path_plt_m=torch.stack(path)	
									path_plt=path_plt_m.detach().cpu().numpy()
									if args.Plot_flag:
										r3d_plot_path(2,path_plt,size_start_end,marker_start,color,color_flag,marker_end,size_generate,paths,path_lengths,size_act,i,j,ax1)
										color_flag=color_flag+1 

							
							
							print("sp_ite"+str(j))
							print(sp)

						if indicator==1:
							path_cost,expert_path_cost,rate=r3d_count_length(path,paths,i,j,path_lengths)
							each_env_sum_rate[i-args.enviro_index_low] += rate
							each_env_count[i-args.enviro_index_low] +=1
							path_cost_env.append(path_cost.item())
							expert_cost_env.append(expert_path_cost)
							print("path",path)
							print("expert_path",paths[i][j][:path_lengths[i][j]])
							print("path_cost:",path_cost,"expert_path_cost:",expert_path_cost,"path_rate:",rate)
				#each path						
							iteration_env.append(iteration)

			# np.set_printoptions(threshold = 1e6)	#	show all the element of matrix		
			if args.Plot_flag:
				r3d_plot_cloud(obs,i,obs_start,ax1)

			# embed_obs=obstacles[i].reshape(len(obstacles[i])//2,2)
			#plt.scatter(embed_obs[:,0], embed_obs[:,1], s=5,c='red')

			#np.set_printoptions(threshold = 1e6)	#	show all the element of matrix	
			#print("obs_recover")
			#print(obs_recover[i])
			# recover_obs=obs_recover[i].reshape(len(obs_recover[i])//2,2) 
			#plt.scatter(recover_obs[:,0], recover_obs[:,1], s=5,c='red')
			iteration_all.append(iteration_env)
			tot.append(et)				
			path_cost_all.append(path_cost_env)
			expert_cost_all.append(expert_cost_env)	
			# with open('tot_'+str(i)+'.csv', 'w', newline='') as file:
			# 	writer = csv.writer(file)
			# 	writer.writerows(tot)
			# with open('path_cost_all'+str(i)+'.csv', 'w', newline='') as file:
			# 	writer = csv.writer(file)
			# 	writer.writerows(path_cost_all)
			# with open('expert_cost_all'+str(i)+'.csv', 'w', newline='') as file:
			# 	writer = csv.writer(file)
			# 	writer.writerows(expert_cost_all)
			# with open('iteration_env'+str(i)+'.csv', 'w', newline='') as file:
			# 	writer = csv.writer(file)
			# 	print("iteration_env",iteration_env)
			# 	writer.writerows(iteration_all)
		
		print("iteration_all!!!!!!!",iteration_all)
		# print("tot",tot)	
		# print("path_cost_all",path_cost_all)
		# print("expert_cost_all",expert_cost_all)
		for i in range (0,args.enviro_index_up-args.enviro_index_low):
			data = np.array(tot[i])	
			mean_value = np.mean(data)
			print("environment:",i+args.enviro_index_low,"mean of time:", mean_value)
			variance_value = np.var(data)
			print("environment:",i+args.enviro_index_low,"variance of time:", variance_value)

			path_len=np.array(path_cost_all[i])	
			print("path_len",path_len)
			mean_value = np.mean(path_len)
			print("environment:",i+args.enviro_index_low,"mean of path:", mean_value)
			variance_value = np.var(path_len)
			print("environment:",i+args.enviro_index_low,"variance of path:", variance_value)

			expert_len=np.array(expert_cost_all[i])	
			print("expert_len",expert_len)
			mean_value = np.mean(expert_len)
			print("environment:",i+args.enviro_index_low,"mean of expert path:", mean_value)
			variance_value = np.var(expert_len)
			print("environment:",i+args.enviro_index_low,"variance of expert path:", variance_value)

			iteration_count=np.array(iteration_all[i])	
			print("iteration_count",iteration_count)
			mean_value = np.mean(iteration_count)
			print("environment:",i+args.enviro_index_low,"iteration_mean:", mean_value)
			variance_value = np.var(iteration_count)
			print("environment:",i+args.enviro_index_low,"iteration_var:", variance_value)



		pickle.dump(tot, open("time_s2D_unseen_mlp.p", "wb" ))

		
		print(enviro_type)
		print("infeasible_path_index1")
		print(infeasible_path_index1)
		print("infeasible_path_index2")
		print(infeasible_path_index2)
		print("invalid_start_or_goal")
		print(invalid_start_or_goal)
		print("success rate_without_replan= %f" % (path_feasible_count/path_count))
		print ("total paths")
		print (tp)
		print ("feasible paths")
		print (fp)
		for i in range (0,args.enviro_index_up-args.enviro_index_low):
			print('env %d path_cost_average_rate is: %f' % (i,each_env_sum_rate[i]/(each_env_count[i]+0.00001)))
		
		if args.Plot_flag:
			plt.show()#!!!!!!1
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--MLP_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--CNN_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--input_size', type=int , default=32, help='dimension of the input vector')
	parser.add_argument('--num_sector_theta', type=int , default=90, help='The number of sectors that divide the space and num_sector must be even number') 
	parser.add_argument('--num_sector_phi', type=int , default=180, help='The number of sectors that divide the space and num_sector must be even number') 
	
	
	parser.add_argument('--dropout_p', type=float , default=0.0, help='The probability of dropout')
	
	parser.add_argument('--Plot_flag', type=int , default=0)
	parser.add_argument('--enviro_index_low', type=int , default=2)
	parser.add_argument('--enviro_index_up', type=int , default=3)
	parser.add_argument('--Path_index_low', type=int , default=2)
	parser.add_argument('--Path_index_up', type=int , default=3)
	parser.add_argument('--Path_lvc_state', type=int , default=0)

	parser.add_argument('--stuck_search_time', type=int , default=0,help='The upper limit of the number of searches for the same point when performing replanning')
	parser.add_argument('--Sparse_index', type=int , default=1,help='Sparse attempt index for possibility search in replanning modules')
	
	


	args = parser.parse_args()
	print(args)
	main(args)


