# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:24:46 2019

@author: caoqi95

Project:
    1. haze remove
        1) Gaussian filtering replaces minimum filtering
        2) Edge preserving filtering replaces minimum filtering
    2. night image 
        1) Invert input image
        2) Haze remove
        3) Invert the haze removal image
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter

def dark_channel(img_rgb, radius):
    
    """
    Calculate the dark channel of a image.
    """
    w, h = img_rgb.shape[0], img_rgb.shape[1]
  
    mapping = np.zeros((w, h), img_rgb.dtype)
    for i in range(w):
        for j in range(h):
            mapping[i, j] = np.min(img_rgb[i, j, :])
    

    mini_filter = cv2.erode(mapping, np.ones((radius, radius)))
    
    return mini_filter


def getA(dark_channel, img_rgb):
    
    """
    Get the Airlight value from the original image and Dark channel.
    """
    
    w, h = dark_channel.shape[0], dark_channel.shape[1]
    light = []
    for i in range(w):
        for j in range(h):
            light.append(dark_channel[i, j])
    
    light.sort()
    light.reverse()
    
    threshold = light[int(0.001*len(light))]
    
    atmosphere = {}
    for i in range(w):
        for j in range(h):
            if dark_channel[i][j] >= threshold:
                atmosphere.update({(i, j):sum(img_rgb[i][j][:])/3.0})
    
    pos = sorted(atmosphere.items(), key=lambda item: item[1], reverse=True)[0][0]
    
    A = int(sum(img_rgb[pos]) / 3.0)
    
    if A > 240:
        A = 240
        
    return A
                
def getTransmission(img_rgb, A, omiga, radius):
    
    """
    Calculate the transmission. 
    """
    
    w, h = img_rgb.shape[0], img_rgb.shape[1]
    mapping = np.zeros((w, h), img_rgb.dtype)
    
    for i in range(w):
        for j in range(h):
            mapping[i, j] = np.min(img_rgb[i, j, :])
    
    # minimum filter
    mini_filter = cv2.erode(mapping, np.ones((radius, radius)))
    return 1 - omiga*mini_filter/A
    
    # gaussion filter 
    #gaussian = cv2.GaussianBlur(mapping, (radius, radius), 0)
    #return 1 - omiga*gaussian/A
    
    # edge-prev - bilateral Filter
    #bi = cv2.bilateralFilter(mapping, radius, 16, 16)
    #return 1 - omiga*bi/A
    
    

def dehaze(img_rgb, A, t, t0=0.1):
    
    """
    Dehaze based on dark channel.
    """
    
    w, h = img_rgb.shape[0], img_rgb.shape[1]
    dehaze = np.zeros(img_rgb.shape, img_rgb.dtype)
    for c in range(img_rgb.shape[-1]):
        for i in range(w):
            for j in range(h):
                dehaze[i, j, c] = int((img_rgb[i, j, c] - A)/max(t[i, j], t0) + A)

    return dehaze
        

if __name__ == "__main__":
    
    # Read img
    path = "D:/CAO_project/cv/project3/haze/(87).jpg"
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dark channel 
    img_dark_c = dark_channel(img_rgb, 3)
    # Get A value
    A = getA(img_dark_c, img_rgb)
    print(A)
    # Transmission
    t = getTransmission(img_rgb, A, 0.95, radius=3)
    # Dehaze
    dehaze = dehaze(img_rgb, A, t)
    # Guided filter
    img_gf = guidedFilter(img_gray, dehaze, radius=12, eps=10, dDepth=-1)

    
    # Plot 
    plt.figure(figsize=(20, 25))
    plt.subplot(4, 2, 1), plt.imshow(img_rgb), plt.title('Original')
    plt.axis("off")
    #plt.subplot(4, 2, 2), plt.imshow(t, cmap="gray"), plt.title('Trans')
    #plt.axis("off")
    #plt.subplot(4, 2, 3), plt.imshow(dehaze), plt.title('dark_channel result')
    #plt.axis("off")
    dehaze = cv2.cvtColor(dehaze, cv2.COLOR_BGR2RGB)
    #cv2.imwrite("D:./result/dark_channel_img_2_edg_pre_2.jpg", dehaze)
    plt.subplot(4, 2, 2), plt.imshow(img_gf), plt.title('After de-haze')
    plt.axis("off")
    img_gf = cv2.cvtColor(img_gf, cv2.COLOR_BGR2RGB)
    #cv2.imwrite("D:./result/haze/img_2_bi_3.jpg", img_gf)
    plt.show()
