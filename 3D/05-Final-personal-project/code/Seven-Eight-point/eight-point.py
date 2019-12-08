# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 00:44:06 2019

@author: User
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

## Loading the data
img1 = cv2.imread('./data/barrsmithA.png') # left imgae
img2 = cv2.imread('./data/barrsmithB.png') # right image

data = np.loadtxt('./data/barrsmith_kps.txt')
#print(data.shape)

n = data.shape[0]
maxvalues = np.max(data[:, :2])
#print(maxvalues)

## RANSAC

max_iterations = np.inf
iterations = 2000
threshold = 3.0
best_inliers = np.array([])
confidence = 0.99
pts_num = 7

method = None
if pts_num == 7:
    method = cv2.FM_7POINT
if pst_num == 8:
    method = cv2.FM_8POINT
    
    
for i in range(iterations):
    indices = np.random.permutation(n)[:pts_num].reshape((1, pts_num))
    #print(indices)
    pts1 = data[indices, :2].reshape((pts_num, 2))
    pts2 = data[indices, 2:4].reshape((pts_num, 2))
    #alphas = data[indices, 4].reshape((8, 1))
    #print(alphas.shape)

    
    # Estimate a fundamental matrix using 8-point algorithm
    F, mask = cv2.findFundamentalMat(pts1,pts2,method)
    #print(F.shape)
    
    if F is None:
        continue
    
    if F.shape[0] != 3:
        continue
    
    # get the inliers
    F = F.reshape((3, 3, -1))
    
    for fi in range(F.shape[-1]):
        inliers = np.array([])
        Fi = F[:, :, fi]
        #print(Fi.shape)
        # Estimate the symmeric epipolar distance for each correspondences
        for j in range(n):
            d1 = data[j, 2:4]#.reshape((1, 2))
            d1 = np.append(d1, 1).reshape((1, -1)) # shape (1, 3)
            l1 = np.dot(d1, Fi)
            
            d2 = data[j, :2] 
            d2 = np.append(d2, 1).reshape((1, -1)).T
            l2 = np.dot(Fi, d2) 
            #print(l2.shape)
            l1 = l1 / np.sqrt(l1[0][0]**2 + l1[0][1]**2)
            l2 = l2 / np.sqrt(l2[0][0]**2 + l2[1][0]**2)
            
            dist = np.abs(np.dot(l1, d2)) + np.abs(np.dot(d1, l2)) * 0.5
            #print(dist)
            if dist < threshold:
                #print(dist)
                inliers = np.append(inliers, j)
            
        if len(best_inliers) < len(inliers):
            # Update inliers of the so-far-the-best model
            best_inliers = inliers
            
            # Update max iteration number
            max_iterations = np.log(1 - confidence) / np.log(1 - (len(best_inliers)/n)**5)
            
    if i > max_iterations:
        break
            
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


# Visualization
print('Number of loaded points: ', n)
print('Number of found inliers: ', len(best_inliers))
print('Number of iterations: ', i)

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img = np.hstack((img1, img2))
plt.figure(figsize=(15, 15))
plt.imshow(img)

for i in range(len(best_inliers)):
    
    color = np.random.random((1, 3))
    
    #plt.plot((data[int(best_inliers[i]), 0], img1.shape[1] + data[int(best_inliers[i]), 2]), 
    #         (data[int(best_inliers[i]), 1], data[int(best_inliers[i]), 3]), color)
    
    plt.scatter(data[int(best_inliers[i]), 0], data[int(best_inliers[i]), 1], s=40, c=color)
    plt.scatter(img1.shape[1] + data[int(best_inliers[i]), 2], data[int(best_inliers[i]), 3], s=40, c=color)
    
plt.axis('off')
plt.savefig('C:/Users/User/Desktop/7-point.png', dpi=200, bbox_inches='tight')
plt.show()

img = np.hstack((img1, img2))
plt.figure(figsize=(15, 15))
plt.imshow(img)

cmap = get_cmap(len(best_inliers))

for i in range(len(best_inliers)):
    
    #color = np.random.random((1, 3))
    
    plt.scatter(data[int(best_inliers[i]), 0], data[int(best_inliers[i]), 1], s=40, c=cmap(i)[:3])
    plt.scatter(img1.shape[1] + data[int(best_inliers[i]), 2], data[int(best_inliers[i]), 3], s=40, c=cmap(i)[:3])
    plt.plot((data[int(best_inliers[i]), 0], img1.shape[1] + data[int(best_inliers[i]), 2]), 
             (data[int(best_inliers[i]), 1], data[int(best_inliers[i]), 3]), c=cmap(i)[:3])
    
plt.axis('off')
plt.savefig('C:/Users/User/Desktop/7-point-2.png', dpi=200, bbox_inches='tight')
#cv2.imwrite("C:/Users/User/Desktop/8-point-2.jpg", img)
plt.show()
