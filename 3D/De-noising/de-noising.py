# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:21:15 2019

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

def denoise(im, U_init, tolerance=0.01, tau=0.125, tv_weight=100):
    
    """ 
    An implementation of the Rudin-Osher-Fatemi (ROF) denoising model 
    using the numerical procedure presented in eq (11) A. Chambolle (2005).
    
    Input: noisy input image (grayscale), initial guess for U, weight of 
    the TV-regularizing term, steplength, tolerance for stop criterion.
    
    Output: denoised and detextured image, texture residual. 
    """
    #im = np.asarray(im)
    m, n = im.shape # size of noisy image
    
    # initialize
    U = U_init
    Px = im # x-component to the dual field
    Py = im # y-conponent of the dual field
    error = 1
    
    while (error > tolerance):
        
        Uold = U
        
        # grandient of primal variable
        GradUx = np.roll(U, -1, axis=1) - U # x-component of U's gradient
        GradUy = np.roll(U, -1, axis=0) - U # y-component of U's gradient
        
        # update the dual varible
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = np.maximum(1, np.sqrt(PxNew**2 + PyNew**2))
        
        Px = PxNew / NormNew # update of x-component (dual)
        Py = PyNew / NormNew # update of y-component (dual)
        
        # update the primal variable
        RxPx = np.roll(Px, 1, axis=1) # right x-translation of x-component
        RyPy = np.roll(Py, 1, axis=0) # right y-translation of y-component
        
        DivP = (Px-RxPx) + (Py - RyPy) # divergence of the dual field
        U = im + tv_weight*DivP # update of the primal variable
        
        # update of error
        error = np.linalg.norm(U - Uold) / np.sqrt(n*m)
        
    return U, im-U # denoised image and texture residual


if __name__ == "__main__":
    """
    
    #img = Image.open("./lena-noise.jpg")   
    img = cv2.imread("./lena-noise.jpg", 0)
    G = filters.gaussian_filter(img, 3)
    U, T = denoise(img, img)
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 3, 2), plt.imshow(U, cmap="gray")
    plt.title("ROF")
    plt.axis("off")
    plt.subplot(1, 3, 3), plt.imshow(G, cmap="gray")
    plt.title("Gaussian(sigma=3)")
    plt.axis("off")
    """
    # create synthetic image with noise
    im = np.zeros((500, 500))
    im[100:400, 100:400] = 128
    im[200:300, 200:300] = 255
    im = im + 30 * np.random.standard_normal((500, 500))
    
    U, T = denoise(im, im)
    G = filters.gaussian_filter(im, 10)
    plt.figure(figsize=(20, 20))
    plt.subplot(3, 1, 1), plt.imshow(im, cmap="gray")
    plt.subplot(3, 1, 2), plt.imshow(G, cmap="gray")
    plt.subplot(3, 1, 3), plt.imshow(U, cmap="gray")
    
    