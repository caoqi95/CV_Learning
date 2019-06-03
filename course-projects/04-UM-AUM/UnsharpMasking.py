# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:48:55 2019

@author: caoqi95
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
def UnsharpMask_Gau(img, Lambda):
    
    low_signal = cv2.GaussianBlur(img, (3, 3), 16)
    high_signal = img - low_signal
    res = Lambda * high_signal + img
    
    return res
"""

def UnsharpMask(img, Lambda):
    
    """# kernel for Unsharp Masking
    kernel = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    """
    # kernel for Laplacian
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    
    high = cv2.filter2D(img, -1, kernel)
    
    res = Lambda * high + img
    
    # Prevent overflow 
    res[res > 255] = 255.0
    res[res < 0] = 0.0
    
    return res

    
if __name__ == "__main__":
    
    path = "D:/CAO_project/cv/project4/um/1.bmp"
    img = cv2.imread(path, 0)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #img_um = UnsharpMask_Gau(img_rgb, 8)
    img_um = UnsharpMask(img, 0.5)
    plt.figure(figsize=(15, 18))
    plt.subplot(2, 2, 1), plt.imshow(img_rgb), plt.title('Original')
    plt.axis("off")
    #plt.subplot(3, 1, 2), plt.imshow(img_um), plt.title('Unsharp Masking 1')
    #plt.axis("off")
    plt.subplot(2, 2, 2), plt.imshow(img_um, cmap="gray"), plt.title('Unsharp Masking')
    plt.axis("off")
    #cv2.imwrite("D:/CAO_project/cv/project4/result/img_um_1.jpg", img_um)
