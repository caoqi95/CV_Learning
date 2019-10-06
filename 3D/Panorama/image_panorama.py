# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:09:38 2019

@author: User
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sift
import warp
import homography

# read images and generate the sift points
featname = ['e' + str(i+1)+'.sift' for i in range(3)]
imname = ['e' + str(i+1)+'.jpg' for i in range(3)]

l = {}
d = {}

for i in range(3):
    sift.process_image(imname[i], featname[i])
    l[i], d[i] = sift.read_features_from_file(featname[i])
    
matches = {}
for i in range(2):
    matches[i] = sift.match(d[i+1], d[i])
    
    
# visualize the matches
for i in range(2):
    im1 = np.array(Image.open(imname[i]))
    im2 = np.array(Image.open(imname[i+1]))
    plt.figure()
    sift.plot_matches(im2,im1,l[i+1],l[i],matches[i],show_below=True)
    
    
# function to convert the matches to homography points
def convert_points(j):
    ndx = matches[j].nonzero()[0]
    fp = homography.make_homog(l[j+1][ndx,:2].T) 
    ndx2 = [int(matches[j][i]) for i in ndx]
    tp = homography.make_homog(l[j][ndx2,:2].T) 
    
    # switch x and y
    fp = np.vstack([fp[1],fp[0],fp[2]])
    tp = np.vstack([tp[1],tp[0],tp[2]])
    return fp,tp

# estimate the homographies
model = homography.RansacModel()

fp, tp = convert_points(0)
H_01 = homography.H_from_ransac(fp, tp, model)[0] # im 0 to 1

fp, tp = convert_points(1)
H_12 = homography.H_from_ransac(fp, tp, model)[0] # im 1 to 2

# warp the images
delta = 800
delta1 = 500

im0 = np.array(Image.open(imname[0]), "uint8")
im1 = np.array(Image.open(imname[1]), "uint8")
im2 = np.array(Image.open(imname[2]), "uint8")

im_01 = warp.panorama(H_01, im0, im1, delta, delta1)
im_12 = warp.panorama(H_12, im1, im2, delta, delta1)
im0 = np.array(Image.open(imname[0]), "f")
im_02 = warp.panorama(np.dot(H_12, H_01), im0, im_12, delta, delta1)

plt.figure(figsize=(15, 15))
im0 = np.array(Image.open(imname[0]), "uint8")
plt.subplot(3, 2, 1), plt.imshow(im0), plt.axis('off'), plt.title('Image 1')
plt.subplot(3, 2, 2), plt.imshow(im1), plt.axis('off'), plt.title('Image 2')
plt.subplot(3, 2, 3), plt.imshow(im2), plt.axis('off'), plt.title('Image 3')
plt.subplot(3, 2, 4), plt.imshow(np.array(im_02, "uint8")), plt.axis('off')
plt.title('Panorama with 3 images')
plt.subplot(3, 2, 5), plt.imshow(np.array(im_01, "uint8")), plt.axis('off')
plt.title('Panorama with Image 1 and Image 2')
plt.subplot(3, 2, 6), plt.imshow(np.array(im_12, "uint8")), plt.axis('off')
plt.title('Panorama with Image 2 and Image 3')
plt.show()
