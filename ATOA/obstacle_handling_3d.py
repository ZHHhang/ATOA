
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy import ndimage
import torch.nn.functional as F
import torch
from ipywidgets import interact, IntSlider
import ipywidgets as widgets
import SimpleITK as sitk



def sample_outer_surface_in_pixel(image): 
    a = F.max_pool3d(image[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1,0,0))[0]
    b = F.max_pool3d(image[None,None].float(), kernel_size=(1,3,1), stride=1, padding=(0,1,0))[0]
    c = F.max_pool3d(image[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0,0,1))[0]
    border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0)
    surface = border - image.float()
    
    return surface.long()

def sample_inner_surface_in_pixel(image): 
    a = F.max_pool3d(-image[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1,0,0))[0]
    b = F.max_pool3d(-image[None,None].float(), kernel_size=(1,3,1), stride=1, padding=(0,1,0))[0]
    c = F.max_pool3d(-image[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0,0,1))[0]
    border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0)
    surface = border + image.float()
    
    return surface.long()



def SamplesFunc(img, value=1, type='array'):
    samples_points = np.where(img == value)
    samples = list(zip(list(samples_points[0]),list(samples_points[1]),list(samples_points[2])))
    if type=='list':
        return samples 
    elif type=='array':
        return np.array(samples)

def SamplesFunc1(img, value=1, type='array', step=2):

    samples_points = np.where(img[::step, ::step, ::step] == value)

    samples_points_adjusted = tuple([points * step for points in samples_points])
    samples = list(zip(*samples_points_adjusted))
    
    if type == 'list':
        return samples
    elif type == 'array':
        return np.array(samples)
    
def surface_to_real(surface_points,bias,scale_para):
    real_points=surface_points/scale_para
    real_points=np.subtract(real_points,bias)
    return real_points
    
