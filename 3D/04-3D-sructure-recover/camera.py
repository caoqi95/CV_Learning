from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt


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