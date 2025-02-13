import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random
from scipy import ndimage

scale_para=4
bias=20
# Environment Encoder
length_obc = np.array([2.0,7.5,5.0,5.0,6.0,4.3,4.5,5.0,3.8,4.5,4.9])
width_obc = np.array([7.0, 5.0,3.0,5.0,5.0,4.7,4.7,6.8,7.0,4.0,4.6])

# def load_dataset(N=30000,NP=1800):
def load_dataset_mask(N=120,NP=1800):

    # obstacles=np.zeros((N,8800),dtype=np.float32)
    # for i in range(0,N):
    #     temp=np.fromfile('../../../DATA_SET/obs_cloud_for_CAE/obc_complex'+str(i)+'.dat')
    #     temp=temp.reshape(len(temp)//2,2) #change/->// due to python3 by zheng
    #     obstacles[i]=temp.flatten()

##########################
    # obc=np.zeros((N,11,2),dtype=np.float32)
    # temp=np.fromfile('../../../DATA_SET/Concave_2D/obs.dat')
    # obs=temp.reshape(len(temp)//2,2) #change by z 

    # temp=np.fromfile('../../../DATA_SET/Concave_2D/obs_perm_concave.dat',np.int32)
    # perm=temp.reshape(167960,11)
    # for i in range(0,N):
    #     for j in range(0,11):
    #         for k in range(0,2):

    #             obc[i][j][k]=obs[perm[i][j]][k]
    # enviro_mask_set=np.zeros((N,int(40*scale_para), int(40*scale_para)))
    # for i in range(0,N):
    #     enviro = np.zeros((int(40*scale_para), int(40*scale_para)))
        
    #     length_obc_mask=length_obc*scale_para
    #     width_obc_mask=width_obc*scale_para
    #     obc_mask=(obc[i]+bias)*scale_para
    #     for j in range(0,11):
    #         center_x, center_y = obc_mask[j]
    #         #Calculate the coordinates of the four corners of the obstacle
    #         left_x = int(np.floor(center_x - length_obc_mask[j] / 2))
    #         right_x = int(np.ceil(center_x + length_obc_mask[j] / 2))
    #         top_y = int(np.floor(center_y - width_obc_mask[j] / 2))
    #         bottom_y = int(np.ceil(center_y + width_obc_mask[j]/ 2))
    #         for x in range(left_x, right_x + 1):
    #             for y in range(top_y, bottom_y + 1):
    #                 # if 0 <= x < enviro.shape[0] and 0 <= y < enviro.shape[1]:  
    #                 enviro[x, y] = 1  
    #     enviro = ndimage.binary_fill_holes(enviro).astype(int) #mask
    #     enviro[0, :] = 0
    #     enviro[-1, :] = 0
    #     enviro[:, 0] = 0
    #     enviro[:, -1] = 0

        # print("enviro",enviro,enviro.shape)
        # enviro.tofile('../../DATA_SET/data_generation/obc_mask_narrow/concave_enviro_' + str(i) + '.dat')


        # print("enviro",enviro,enviro.shape)
        # enviro.tofile('../../../DATA_SET/Concave_2D/obc_mask_concave/concave_enviro_' + str(i) + '.dat')
        # file_path = '../../../DATA_SET/Concave_2D/obc_mask_concave/concave_enviro_' + str(i) + '.dat'  
        # shape = (int(40 * scale_para), int(40 * scale_para))

        # enviro_loaded = np.fromfile(file_path, dtype=int).reshape(shape)

        # print("Loaded enviro for index", i, ":", enviro_loaded)
        # print("Shape:", enviro_loaded.shape)
        # print("enviro_loaded-enviro:", enviro_loaded-enviro)
        # if np.all(enviro_loaded == enviro):
        #     print("yes")
        # else:
        #     print("wrong")
        #     assert 0 
##########################  
    enviro_mask_set=np.zeros((N,int(40*scale_para), int(40*scale_para)))
	
    for i in range(0,N):   
        file_path = '../../../DATA_SET/Concave_2D/obc_mask_concave_narrow/concave_enviro_' + str(i) + '.dat'  
        shape = (int(40 * scale_para), int(40 * scale_para))
        enviro = np.fromfile(file_path, dtype=int).reshape(shape)   
        enviro_mask_set[i]=enviro
    # assert 0
    # print("enviro_mask_set",enviro_mask_set,enviro_mask_set.shape,type(enviro_mask_set))
    # assert 0
    return 	enviro_mask_set	