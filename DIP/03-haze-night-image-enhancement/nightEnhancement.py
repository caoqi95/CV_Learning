# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:19:06 2019

@author: caoqi95
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter

def invert(img_rgb):
    """
    Invert the image.
    """
    
    w, h, depth = img_rgb.shape
    invert = np.zeros_like(img_rgb)
    for c in range(depth):
        invert[:, :, c] = 255 - img_rgb[:, :, c]
    
    return invert
 
def dark_channel(img_rgb, radius):
    
    """
    Calculate the dark channel of a image.
    """
    """
    mapping = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            mapping[i, j] = np.min(img_rgb[i, j, :])
    
    

    mini_filter = cv2.erode(mapping, np.ones((radius, radius)))
    """
    b, g, r = cv2.split(img_rgb)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = np.ones((radius, radius))
    mini_filter = cv2.erode(dc, kernel)
    
    return mini_filter

def getA(dark_channel, img_rgb):
    
    """
    Get the 'A' value from the original image and Dark channel.
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
    
    """
    If you use Guided Filter work on the inverted image the result will be better.
    """   
    # guided Filter
    #guided = guidedFilter(img_rgb, mapping, radius=20, eps=60, dDepth=-1)
    #return 1 - omiga*guided/A
    

def dehaze(img_rgb, A, t, t0=0.1):
    
    """
    Dehaze based on dark channel
    """
    
    w, h = img_rgb.shape[0], img_rgb.shape[1]
    dehaze = np.zeros(img_rgb.shape, img_rgb.dtype)
    for c in range(img_rgb.shape[-1]):
        for i in range(w):
            for j in range(h):
                dehaze[i, j, c] = int((img_rgb[i, j, c] - A)/max(t[i, j], t0) + A)

    return dehaze

        
if __name__ == "__main__":
    
    path = "D:/CAO_project/cv/project3/night/(69).jpg"
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = invert(img_rgb)

    # Dark channel 
    img_dark_c = dark_channel(img_invert, 3)
    # A value
    A = getA(img_dark_c, img_invert)
    print(A)
    # Transmission
    t = getTransmission(img_invert, A, 0.8, 3)
    # Recover and Invert
    recover = dehaze(img_invert, A, t)
    img_re = invert(recover)
    
    """
    If you use Guided Filter work on the inverted image the result will be better.
    """
    #img_gf = guidedFilter(img_gray, img_re, radius=12, eps=10, dDepth=-1)
    
    plt.figure(figsize=(15, 20))
    plt.subplot(4, 2, 1), plt.imshow(img_rgb), plt.title('Original')
    plt.axis("off")
    #plt.subplot(4, 2, 2), plt.imshow(img_invert), plt.title('Invert')
    #plt.subplot(4, 2, 3), plt.imshow(recover), plt.title('Recover')
    plt.subplot(4, 2, 2), plt.imshow(img_re), plt.title('After processed')
    plt.axis("off")
    img_re = cv2.cvtColor(img_re, cv2.COLOR_BGR2RGB)
    #cv2.imwrite("D:./result/night/img_3_15.jpg", img_re)

