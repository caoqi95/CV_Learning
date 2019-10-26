# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:48:27 2019

@author: User
"""

from PIL import Image
from OpenGL.GLUT import *
import numpy as np
from scipy import linalg
import homography
import sift
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame, pygame.image
from pygame.locals import *
import glob 

def set_projection_from_camera(K):
    
    """
    Set view from a camera calibration matrix.
    """
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    fx = K[0, 0]
    fy = K[1, 1]
    fovy = 2*np.arctan(0.5*height/fy)*180/np.pi
    aspect = (width*fy)/(height*fx)
    
    # define the near and far clipping planes
    near = 0.1
    far = 100.0
    
    # set perspective
    gluPerspective(fovy, aspect, near, far)
    glViewport(0, 0, width, height)

def set_modelview_from_camera(Rt):
    
    """
    Set the model view matrix from camera pose.
    """
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # rotate teapot 90 deg around x-axis so that z-axis is up
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    
    # set rotation to best approximation
    R = Rt[:, :3]
    U, S, V = linalg.svd(R)
    R = np.dot(U, V)
    R[0, :] = -R[0, :] # change sign of x-axis
    
    # set translation
    t = Rt[:, 3]
    
    # set 4*4 model view matrix
    M = np.eye(4)
    M[:3, :3] = np.dot(R, Rx)
    M[:3, 3] = t
    
    # transpose and flatten to grt column order
    M = M.T
    m = M.flatten()
    
    # replace model view with new matrix
    glLoadMatrixf(m)

def draw_background(frame):
    
    """
    Draw background image using a quad.
    """
    
    # load backgound image (should be .bmp) to OpenGL texture
    bg_image = pygame.image.load(frame).convert()
    bg_data = pygame.image.tostring(bg_image, "RGBX", 1)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #tx_image = cv2.flip(frame, 0)
    #tx_image = Image.fromarray(tx_image)
    #ix = tx_image.size[0]
    #iy = tx_image.size[1]
    #tx_image = tx_image.tobytes('raw', 'RGBX', 0, -1)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # bind the texture
    glEnable(GL_TEXTURE_2D)
    #texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    
    # create quad to fill the whole window
    #glBindTexture(GL_TEXTURE_2D, texture_id)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, -1.0)
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, -1.0)
    glEnd()
    
    # clear the texture
    glDeleteTextures(1)

def draw_teapot(size):
    
    """
    Draw a red teapot at the origin.
    """
    glEnable(GL_LIGHTING) 
    glEnable(GL_LIGHT0) 
    glEnable(GL_DEPTH_TEST) 
    glClear(GL_DEPTH_BUFFER_BIT)
    
    # draw red teapot
    glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0]) 
    glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0]) 
    glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0]) 
    glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0) 
    glutSolidTeapot(size)


class Camera(object):
    """
    Class for representing pin-hole cameras.
    """
    
    def __init__(self, P):
        
        """
        Initialize P = K[R|t] camera model.
        """
        
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation
        self.x = None # camera center
        
    def project(self, x):
        
        """
        Project points in X (4*n array) and normalize coordinates.   
        """
        
        x = np.dot(self.P, x)
        for i in range(3):
            x[i] /= x[2]
        return x
    
    def factor(self):
        
        """
        Factorize the camera matrix into K,R,t as P = K[R|t].
        """
        
        # factor first 3*3 part
        K, R = linalg.rq(self.P[:, :3])
        
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) < 0:
            T[1, 1] *= -1
        
        self.K = np.dot(K, T)
        self.R = np.dot(T, R) # T is its own inverse
        self.t = np.dot(linalg.inv(self.K), self.P[:, 3])
        
        return self.K, self.R, self.t
    
    def center(self):
        
        """
        Compute and return the camera center.
        """
        
        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -np.dot(self.R.T, self.t)
            return self.c
        
def rotation_matrix(a):
    
    """
    Creates a 3D rotation matrix for rotation
    around the axis of the vector a.
    """
    
    R = np.eye(4)
    R[:3, :3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

    return R

def my_calibration(sz):
    
    """
    Function for camera calibration
    """
    
    row, col = sz
    fx = 2917*col/3024 # parameters get from the book's method
    fy = 2800*row/4032
    K = np.diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K

def get_H(im0_path, im1_path):
    
    """
    Get the Homography matrix.
    """
    # compute features
    sift.process_image(im0_path,  'im0.sift')
    l0, d0 = sift.read_features_from_file('im0.sift')

    sift.process_image(im1_path, 'im1.sift')
    l1, d1 = sift.read_features_from_file('im1.sift')

    # match features and estimate homography
    matches = sift.match_twosided(d0, d1)
    ndx = matches.nonzero()[0]
    fp = homography.make_homog(l0[ndx, :2].T)
    ndx2 = [int(matches[i]) for i in ndx]
    tp = homography.make_homog(l1[ndx2, :2].T)

    model = homography.RansacModel()
    H = homography.H_from_ransac(fp, tp, model)[0]
    
    return H

def cube_points(c, wid):
    
    """
    Create a list of points for plotting a cube
    with plot.(the first 5 points are the bottom square, some sides repeated).
    """
    p = []
    # bottom
    p.append([c[0]-wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]-wid, c[2]-wid]) # same as first to close plot
    
    # top
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]-wid, c[2]+wid]) # same as first to close plot
    
    # vertical sides
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    
    return np.array(p).T

def get_camera_params(sz, H):
    
    """
    Get camere parametes: K, Rts.
    """
    # camera calibration
    K = my_calibration((sz))
    #print(K)
    # 3D points at plane z=0 with sides of length 0.2
    box = cube_points([0, 0, 0.1], 0.1)

    # project bottom square in first image
    cam1 = Camera(np.hstack((K, np.dot(K, np.array([[0], [0], [-1]])))))

    # first points are the bottom square
    box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

    # use H to transfer points to the second image
    box_trans = homography.normalize(np.dot(H, box_cam1))

    # compute second camera metrx from cam1 and H
    cam2 = Camera(np.dot(H, cam1.P))
    A = np.dot(linalg.inv(K), cam2.P[:, :3])
    A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])]).T
    cam2.P[:, :3] = np.dot(K, A)
    
    Rt = np.dot(linalg.inv(K), cam2.P)
    
    return K, Rt



width, height = 960, 544

def setup():
    
    """
    Setup window and pygame environment.
    """
    
    pygame.init()
    pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')
    

# get frames' paths
paths = glob.glob('C:/Users/User/Desktop/course/3D/project/AR-system/book2/*')


i = 10 # used for get the ith frame
im0_path = paths[0] # get the first frame
im1_path = paths[0+i]

# get H and camere parameters
H = get_H(im0_path, im1_path)
K, Rt = get_camera_params((544, 960), H)

# implement AR
setup()
draw_background(paths[0+i])
set_projection_from_camera(K)
set_modelview_from_camera(Rt)
draw_teapot(0.05)
pygame.display.flip()
    
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
            pygame.quit()


