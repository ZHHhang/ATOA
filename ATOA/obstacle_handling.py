
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy import ndimage
import torch.nn.functional as F
import torch

def sample_outer_surface_in_pixel(image): 
    a = F.max_pool2d(image[None,None].float(), kernel_size=(3,1), stride=1, padding=(1,0))[0]
    b = F.max_pool2d(image[None,None].float(), kernel_size=(1,3), stride=1, padding=(0,1))[0]
    border, _ = torch.max(torch.cat([a,b],dim=0),dim=0) 
    surface = border - image.float()
    
    return surface.long()

def sample_inner_surface_in_pixel(image): 
    a = F.max_pool2d(-image[None,None].float(), kernel_size=(3,1), stride=1, padding=(1,0))[0]
    b = F.max_pool2d(-image[None,None].float(), kernel_size=(1,3), stride=1, padding=(0,1))[0]
    border, _ = torch.max(torch.cat([a,b],dim=0),dim=0) 
    surface = border + image.float()
    
    return surface.long()



def SamplesFunc(img, value=1, type='array'):

    samples_points = np.where(img == value)
    samples = list(zip(list(samples_points[0]),list(samples_points[1])))
    # print("samples")
    # print(samples)

    if type=='list':
        return samples 
    elif type=='array':
        return np.array(samples)
    
def surface_to_real(surface_points,bias,scale_para):
    real_points=surface_points/scale_para
    real_points=np.subtract(real_points,bias)
    return real_points
    








