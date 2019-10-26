# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:32:51 2019

@author: User
"""
# -*- coding:utf8 -*-
import cv2
import os
import shutil
 
# filename
filename = 'book2.mp4'
 
# path for save the frames
savedpath = filename.split('.')[0] + '/'
isExists = os.path.exists(savedpath)
if not isExists:
    os.makedirs(savedpath)
    print('path of %s is build'%(savedpath))
else:
    shutil.rmtree(savedpath)
    os.makedirs(savedpath)
    print('path of %s already exist and rebuild'%(savedpath))
    
 
# fps
fps = 24
# interval
count = 6
 
# read the video
videoCapture = cv2.VideoCapture(filename)
i = 0
j = 0
 
while True:
    success,frame = videoCapture.read()
    i+=1
    if(i % count ==0):
        # save as image
        j += 1
        savedname = filename.split('.')[0]  + '_' + str(i)+'.jpg'
        cv2.imwrite(savedpath + savedname ,frame)
        print('image of %s is saved'%(savedname))
    if not success:
        print('video is all read')
        break
    
