# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:21:15 2019

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

def denoise(im, U_init, tv_weight, tolerance=0.01, tau=0.125):
    
    """ 
    An implementation of the Rudin-Osher-Fatemi (ROF) denoising model 
    using the numerical procedure presented in eq (11) A. Chambolle (2005).
    
    Input: 
        - im: noisy input image (grayscale), 
        - U_init: initial guess for U, 
        - tv_weight: weight of the TV-regularizing term, 
        - tolerance: tolerance for stop criterion,
        - tau: steplength.
    
    Output: 
        denoised and detextured image, texture residual. 
    """
    im = np.asarray(im)
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
        
        # update of error-measure
        error = np.linalg.norm(U - Uold) / np.sqrt(n*m)
        
    return U, im-U # denoised image and texture residual


if __name__ == "__main__":
    
    img = cv2.imread("D:/CAO_project/lena-noise.jpg", 0)
    G = filters.gaussian_filter(img, 5)
    U, T = denoise(img, img, 100)
    plt.figure(figsize=(12, 12))
    plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray")
    plt.title("Noisy image")
    plt.axis("off")
    plt.subplot(1, 3, 2), plt.imshow(U, cmap="gray")
    plt.title("ROF(TV_wts=100)")
    plt.axis("off")
    plt.subplot(1, 3, 3), plt.imshow(G, cmap="gray")
    plt.title("Gaussian(sigma=5)")
    plt.axis("off")
    plt.show()
