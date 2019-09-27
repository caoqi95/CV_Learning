"""
some algorithms which based on the histogram equalization.

@author: caoqi95
@time: 2019-03-20
@github: https://github.com/caoqi95
@email: caoqi95@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

class enhanceContrast(object):
    
    def __init__(self):
        pass
    
    """
    Baisc Histogram Equalization
    """

    def basic_HE(self, img):
        
        img_eq = cv2.equalizeHist(img)
        
        return img_eq

    """
    Bi-Histogram Equalization(BHE)
    """
        
    def BHE(self, img):
    
        # image mean
        img_mean = int(np.mean(img))
    
        # get two subimages
        img_l = img.flatten().compress((img.flatten() <= img_mean).flat)
        img_u = img.flatten().compress((img.flatten() > img_mean).flat)
        
        # cdf of low subimage
        hist_l, bins_l = np.histogram(img_l, img_mean+1, [0, img_mean])
        pdf_l = hist_l / np.prod(img_l.size)
        cdf_l = pdf_l.cumsum()

        # transform func of low
        cdf_l = cdf_l *(img_mean - img.min()) + img.min()          
    
    
        # cdf of upper subimage
        hist_u, bins_u = np.histogram(img_u, 256-img_mean, [img_mean+1, 256])
        pdf_u = hist_u / np.prod(img_u.size)
        cdf_u = pdf_u.cumsum()

        # transform func of upper
        cdf_u = cdf_u *(img.max() - (int(img_mean) + 1)) + (int(img_mean) + 1)
    
        cdf_new = np.concatenate((cdf_l, cdf_u))
        new_img = cdf_new[img.ravel()]
        img_eq = np.reshape(new_img, img.shape)
    
        return img_eq
    
    """
    Clipped Histogram Equalization
    """
    
    def clipped_HE(self, img, limit):

        hist, bins = np.histogram(img.ravel(), 256, [0, 256])
        #pdf = hist / np.prod(img.shape)
        #cdf = hist.cumsum()
    
        new_hist = np.zeros((256))

        for i in range(0, 256):
            new_hist[i] = np.clip(hist[i], 0, limit)
    
        new_pdf = new_hist / np.prod(img.shape)
        new_cdf = new_pdf.cumsum()
        new_cdf = new_cdf * 255
        img_eq = new_cdf[img]
    
        return img_eq
    
    """
    Bi-Histogram Equalization with a Plateau Limit(BHEPL)
    """
    def BHEPL(self, img):
    
        # image mean
        img_mean = int(np.mean(img))
    
        # getting two subimages
        img_l = img.flatten().compress((img.flatten() <= img_mean).flat)
        img_u = img.flatten().compress((img.flatten() > img_mean).flat)
    
        hist_l, bins_l = np.histogram(img_l.flatten(), img_mean, [0, img_mean])
        t_l = hist_l.sum() / (img_mean + 1)
    
        hist_u, bins_u = np.histogram(img_u.flatten(), 256-img_mean, [img_mean+1, 256])
        t_u = hist_u.sum() / (255 - img_mean)
    
        h_cl = np.zeros(len(hist_l))
        for i in range(len(hist_l)):
            if hist_l[i] <= t_l:
                h_cl[i] = hist_l[i]
            else:
                h_cl[i] = t_l
            
        h_ul = np.zeros(len(hist_u))
        for i in range(len(hist_u)):
            if hist_u[i] <= t_u:
                h_ul[i] = hist_u[i]
            else:
                h_ul[i] = t_u
    
        M1 = h_cl.sum()
        M2 = h_ul.sum()
    
    
        pdf_l = hist_l / M1
        pdf_u = hist_u / M2
    
        cdf_l = pdf_l.cumsum()
        cdf_u = pdf_u.cumsum()
    
        new_cdf_l = (img_mean - img.min()) * (cdf_l - 0.5*pdf_l) + img.min()
        new_cdf_u = (img.max() - img_mean - 1) * (cdf_u - 0.5*pdf_u) + img_mean + 1
    
        new_cdf = np.concatenate((new_cdf_l, new_cdf_u))
    
        img_eq = new_cdf[img]
    
        return img_eq

    """
    Adaptive Histogram Equalization
    """
    
    def adaptive_HE(self, img):
    
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_eq = clahe.apply(img)
    
        return img_eq
