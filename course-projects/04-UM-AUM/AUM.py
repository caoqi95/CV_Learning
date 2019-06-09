# -*- coding: utf-8 -*-
"""
Modified on Thu May 30 21:44:30 2019

@author: caoqi95

Original version from this Repo:
https://github.com/YueHazelZheng/scikit-image/blob/master/skimage/filters/_unsharp_mask.pyx
"""
import math
import numpy as np
from scipy import signal
from scipy import ndimage

import cv2
import matplotlib.pyplot as plt


def precision_image(auto_corr, input_arr, Alpha):
    
    Precision = np.empty_like(Alpha)
    rowNum, colNum = auto_corr.shape[0], auto_corr.shape[1]
    
    for i in range(rowNum):
        for j in range(colNum):
            a, b = auto_corr[i, j, 0, 0], auto_corr[i, j, 0, 1]
            d = auto_corr[i, j, 1, 1]
            
            x, y = input_arr[i, j, 0], input_arr[i, j, 1]
            
            Precision[i, j, 0] = 1 / (a * d - b**2) * (d * x - b * y)
            Precision[i, j, 1] = 1 / (a * d - b**2) * (-b * x + a * y)
            
    return Precision

"""
def local_var(img):
    
    
    #Calculate the local variance for all pixels in the image.
    
    var = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #for c in range(img.shape[2]):
            input_arr = img[i:i+3, j:j+3]
            mean = np.sum(input_arr) / 9
            var_i = round(np.sum((input_arr - mean) * (input_arr - mean))/9, 2)
            var.append(var_i)
    var = np.array(var).reshape(img.shape)
    
    return var
"""
def local_var(input_arr_1d):
    
    """
    Calculate the local variance for a pixel in the image.
    """
    
    #print("Input shape:", input_arr_1d.shape)
    input_arr = input_arr_1d.reshape(3, 3)
    
    sum1 = 0
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            sum1 += input_arr[i, j]
            
    mean = sum1 / 9
    
    sum2 = 0
    
    for i in range(3):
        for j in range(3):
            sum2 += (input_arr[i, j] - mean) * (input_arr[i, j] - mean)
    
    var = sum2 / 9
    
    return var
            
def gain(img_var, t1, t2, dl, dh, d=1):
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            val = img_var[i, j]
            
            if val < t1:
                out = d
            if val >= t2:
                out = dl
            else:
                out = dh
                    
            img_var[i, j] = out
    
    return img_var

def auto_correlation(beta, G):
    """
    Calculate the estimate of the auto correlation matrix of G.
    """
    m, n  = G.shape[0], G.shape[1]
   
    ret_mtx = np.zeros((m, n, 2, 2))
    
    ret_mtx[:, 0] = np.identity(2)
    
    Gcorr = np.multiply(G[..., np.newaxis], G[..., np.newaxis, :])

    for j in range(1, n):
        ret_mtx[:, j] = np.multiply((1 - beta), ret_mtx[:, j-1]) + np.multiply(beta, Gcorr[:, j])
        
    return ret_mtx

def adaptive_unsharp_mask(img, t1=20, t2=150, dh=3, dl=2, d=3, mu=0.1, beta=0.5, iternum=2000):
    
    img = img.astype('float64')
    m, n = img.shape[0], img.shape[1]
    
    g_filter = np.array([[-1, -1, -1],
                     [-1, 8, -1], 
                     [-1, -1, -1]])

    h_filter = np.array([[0, 0, 0], 
                     [-1, 2, -1], 
                     [0, 0, 0]])

    v_filter = np.array([[0, -1, 0],
                     [0, 2, 0], 
                     [0, -1, 0]])    

    # apply linear high pass on image to obtain local dynamics
    input_dynamics = signal.convolve(img, g_filter, mode='same')


    # applt horizontal and vertical Laplacian operatoe on image
    # to obtain correction signal
    correction_x = signal.convolve(img, h_filter, mode='same')
    correction_y = signal.convolve(img, v_filter, mode='same')
    
    img_var = ndimage.generic_filter(img, local_var, size=3)

    #print("min:", img_var.min())
    #print("mean:", img_var.mean())
    #print("max:", img_var.max())
    variance_gain = gain(img_var, t1, t2, dh, dl)

    # desired activity level in the output image
    desired_dynamics = np.multiply(variance_gain, input_dynamics)
    

    # apply linear highpass filter
    g_zx = signal.convolve(correction_x, g_filter, mode='same')
    g_zy = signal.convolve(correction_y, g_filter, mode='same')
    

    G = np.dstack((g_zx, g_zy))

    R = auto_correlation(beta, G)

    # initialize scaling vector for correction signal
    Alpha = np.zeros((m, n, 2))
    Precision = precision_image(R, G, Alpha)

    error_norm_prev = np.infty
    desired_error = np.subtract(desired_dynamics, input_dynamics)
    
    
    # update 
    n = 0
    for k in range(iternum):
        
        error = desired_error.copy()
        
        error -= np.sum(Alpha * G, axis=2)
        #error -= np.einsum('ijk, ijk->ij', Alpha, G)
        error_norm = np.linalg.norm(error)
        
        if math.pow(np.linalg.norm(error) - error_norm_prev, 2) < 1e-8:
            break
        
        error_norm_prev = error_norm
        
        Alpha[:, 1:] = (Alpha + 2*mu*np.multiply(error[..., np.newaxis], Precision))[:, 1:]
        n +=1
    print(n)
    print("error: ", np.abs(np.linalg.norm(error) - error_norm_prev))
    
    output_img = img + np.multiply(Alpha[..., 0], correction_x) + \
        np.multiply(Alpha[..., 1], correction_y)

    output_img[output_img > 255] = 255
    output_img[output_img < 0] = 0
    
    
    return output_img.astype('uint8')
            
if __name__ == "__main__":
    
    path = "D:/CAO_project/cv/project4/um/1.bmp"
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_um = adaptive_unsharp_mask(img_gray, t1=30, t2=200, dh=7, dl=3, d=1, mu=0.5, beta=0.8, iternum=1000)

    plt.figure(figsize=(15, 18))
    plt.subplot(2, 2, 1), plt.imshow(img_rgb), plt.title('Original')
    plt.axis("off")
    plt.subplot(2, 2, 2), plt.imshow(img_um, cmap="gray"), plt.title('Adaptive Unsharp Masking')
    plt.axis("off")
    #cv2.imwrite("D:/CAO_project/cv/project4/result/img_aum_1.jpg", img_um)
    
