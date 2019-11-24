# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:42:00 2019

@author: User
"""

import homography
import sfm
import sift
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# calibration
#K = np.array([[2394, 0, 932], [0, 2398, 628], [0, 0, 1]])
#K = np.array([[1253, 0, 866], [0, 2137, 1155], [0, 0, 1]])
K = np.array([[1670, 0, 1155], [0, 1604, 866], [0, 0, 1]])

# load images and compute features
#im1_path = 'C:/Users/User/Desktop/course/3D/project/pcv_data/alcatraz1.jpg'
#im2_path = 'C:/Users/User/Desktop/course/3D/project/pcv_data/alcatraz2.jpg'
im1_path = './f1.jpg'
im2_path = './f2.jpg'

im1 = np.array(Image.open(im1_path))
sift.process_image(im1_path, 'im1.sift')
l1, d1 = sift.read_features_from_file('im1.sift')

im2 = np.array(Image.open(im2_path))
sift.process_image(im2_path, 'im2.sift')
l2, d2 = sift.read_features_from_file('im2.sift')

# match features
matches = sift.match_twosided(d1, d2)
ndx = matches.nonzero()[0]

# make homogeneous and normalize with inv(K)
x1 = homography.make_homog(l1[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(l2[ndx2, :2].T)

x1n = np.dot(np.linalg.inv(K), x1)
x2n = np.dot(np.linalg.inv(K), x2)

# estimate E with RANSAC
model = sfm.RansacModel()
E, inliers = sfm.F_from_ransac(x1n, x2n, model)

# compute camera matrices (P2 will be list of four solutions)
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_essential(E)


# pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
    # triangulate inliers and compute depth for each camera
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
    d1 = np.dot(P1, X)[2]
    d2 = np.dot(P2[i], X)[2]
    if np.sum(d1>0) + np.sum(d2>0) > maxres:
        maxres = np.sum(d1>0) + np.sum(d2>0)
        ind = i
        infront = (d1>0) & (d2>0)
        
# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[ind])
X = X[:, infront]

# 3D plot
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(-X[0], X[1], X[2], 'k.')
ax.axis('off')
#plt.axis('off')

# plot the projection of X

import camera

# project 3D points
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[ind])
x1p = cam1.project(X)
x2p = cam2.project(X)
"""
Xt = sfm.triangulate(x1p[:, inliers], x2p[:, inliers], P1, P2[ind])
Xt = Xt[:, infront]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(-Xt[0], Xt[1], Xt[2], 'k.')
ax.axis('off')
plt.show()
"""
# reverse K normalization
x1p = np.dot(K,x1p)
x2p = np.dot(K,x2p)


plt.figure()
plt.imshow(im1)
plt.gray()
plt.plot(x1p[0],x1p[1],'b.')
plt.plot(x1[0],x1[1],'r.')
plt.axis('off')

plt.figure()
plt.imshow(im2)
plt.gray()
plt.plot(x2p[0],x2p[1],'b.')
plt.plot(x2[0],x2[1],'r.')
plt.axis('off')
plt.show()
