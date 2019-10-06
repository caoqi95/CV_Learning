# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:09:38 2019

@author: User
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """
    Process an image and save the results in a file.
    """
    path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
    path = path+"\\Panorama\\utils\\win32vlfeat\\sift.exe "
    #print(path)
    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
    
    cmmd = str(path + imagename + " --output=" + resultname
              + " " + params)
    
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)
    
def read_features_from_file(filename):
    
    f = np.loadtxt(filename)
    return f[:, :4], f[:, 4:]

def write_features_to_file(filename, locs, desc):
    
    np.savetxt(filename, np.hsatck((locs, desc)))
    
    
def match(desc1, desc2):
    
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])
    
    dist_ratio = 0.6
    desc1_size = desc1.shape
    
    matchscores = np.zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T # precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i, :], desc2t)
        dotprods = 0.9999 * dotprods
        # inverse consine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))
        
        # check if nearest neighbor has angle less than dist_radio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
            
    return matchscores

def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.
    
    return np.concatenate((im1,im2), axis=1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches
        input: im1,im2 (images as arrays), locs1,locs2 (location of features), 
        matchscores (as output from 'match'), show_below (if images should be shown below). """
    
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = np.vstack((im3,im3))
    
    # show image
    plt.imshow(im3)
    
    # draw lines for matches
    cols1 = im1.shape[1]
    for i in range(len(matchscores)):
        if matchscores[i] > 0:
            plt.plot([locs1[i][0],locs2[matchscores[i, 0], 0]+cols1],[locs1[i][1],locs2[matchscores[i, 0], 1]],'c')
    plt.axis('off')
