# This script is an example to calculate smoothness metrics for trajectory data stored in CSV files.
# It includes functions to handle string data, remove outliers, compute Dimensionless Squared Jerk (DSJ),
# squared differences, and mean curvature. The results are printed with their means and standard deviations.    

import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d
import re

def deal_str(data,reshape=True): # Handle string format data and convert to float lists
    if '['  in data.iloc[0]:
        reshape=False
    if reshape:
        total = []
        data = data.dropna()
        x,y,z=data['0'].tolist(),data['1'].tolist(),data['2'].tolist()
        return x,y,z  
    else:
        total = []
        for i in range(len(data)):
            for j in data.iloc[i]:
                match = re.search(r"\[([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\]", j)
                if match:
                    x,y,z = match.groups()
                    total.append([float(x), float(y), float(z)])
        return subl(total,0),subl(total,1),subl(total,2)
    
def subl(li,idx=0): # Extract a specific index from a list of lists
    li=[sublist[idx] for sublist in li if len(sublist)>0]
    return li
def sort_index(li):# Sort file names based on the trailing number
    sorted_paths = sorted(li,key=lambda x: int(x.split('/')[-1].split('tra')[1].split('.')[0]))
    return sorted_paths

def remove_outliers_iqr(data):# Using IQR method to remove outliers
    data = np.array(data)
    Q1 = np.percentile(data, 10)
    Q3 = np.percentile(data, 90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data >= float(lower_bound)) & (data <= float(upper_bound))]

    return filtered_data

def remove_outliers_std(data, threshold=10): #  Using Standard Deviation method to remove outliers
    data = np.array(data)

    mean = np.mean(data)
    std_dev = abs(np.std(data))

    lower_bound = mean - threshold * std_dev
    upper_bound = mean + threshold * std_dev
    
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_data

def calculate_dsj(x, y, z, time_step):# Calculate Dimensionless Squared Jerk (DSJ)
    """
    DSJ
    """
    t = np.arange(len(x)) * time_step
    t1 = t[0]
    t2 = t[-1]


    vx = np.gradient(x) / time_step
    vy = np.gradient(y) / time_step
    vz = np.gradient(z) / time_step

    ax = np.gradient(vx) / time_step
    ay = np.gradient(vy) / time_step
    az = np.gradient(vz) / time_step

    jx = np.gradient(ax) / time_step
    jy = np.gradient(ay) / time_step
    jz = np.gradient(az) / time_step

    j_squared = jx**2 + jy**2 + jz**2

    integral_j_squared = np.trapz(j_squared, t)
    
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    path_length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    
    dsj = integral_j_squared * (t2 - t1)**5 / (path_length**2)
    return dsj


def gen_metri(pos): # Generate metrics: DSJ, squared difference, mean curvature
    
    li=[]

    dt = 0.01
    x,y,z=deal_str(pos)

    sigma=3 # Gaussian smoothing parameter
    x_smooth = gaussian_filter1d(x, sigma)
    y_smooth = gaussian_filter1d(y, sigma)
    z_smooth = gaussian_filter1d(z, sigma) 
    
    dsj=calculate_dsj(x_smooth,y_smooth,z_smooth,dt)
    squared_diff=np.sqrt((x - x_smooth)**2 + (y - y_smooth)**2+ (z - z_smooth)**2)
    

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)
    v = np.stack((dx, dy, dz), axis=1) 

    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)
    ddz = np.gradient(dz, dt)
    a = np.stack((ddx, ddy, ddz), axis=1)  


    cross_product = np.cross(v, a)
    curvature = np.linalg.norm(cross_product, axis=1) / (np.linalg.norm(v, axis=1) ** 3 + 1e-8)

    if np.mean(squared_diff)>1: # remove outliers if squared_diff too large
        data=[dsj,squared_diff,curvature]
        re=[]
        for i in data:
            re.append(remove_outliers_iqr(i))
        dsj,squared_diff,curvature=re
    li.extend([np.mean(dsj),np.mean(squared_diff),np.mean(curvature)])
    return li

def metrics(file_path,exclude=None,pri=True): # Calculate metrics for all files in a directory
    paths=[file_path+'/'+path for path in sort_index(os.listdir(file_path))]
    result=[]
    for i,path in enumerate(paths):
        if exclude is not None and i in exclude:
            continue
        else:
            file=pd.read_csv(path)
            metri=gen_metri(file)
            result.append(metri)
    m = {
        "dsj":subl(result,0),
        "squared_diff":subl(result,1),
        "mean_curvature": subl(result,2),
    }
    if pri:
        printmeanstd(m)
    return m   
def printmeanstd(li): # Print mean and standard deviation for each metric
    for key,values in li.items():
        if key=='dsj' or key=='mean_curvature':
            print(f'{key}: mean={np.mean(values)}, std={float(np.std(values))}')
            print('\n')

lpc,lpck=metrics('./data/ttra/can_ph_cnn'),metrics('./data/ttra/can_ph_cnn_k')