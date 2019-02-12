"""
Canny edge detection algorithm implement by Python3

@author: caoqi95
@time: 2019-02-12
@github: https://github.com/caoqi95
@email: caoqi95@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

class Canny(object):

    def __init__(self):
        pass

    # Step 0: Get greyed
    def gray(self, img_path):
        """
        Calculate function:
        Gray(i,j) = [R(i,j) + G(i,j) + B(i,j)] / 3
        or :
        Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
        """
        img = plt.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return np.dot(img_rgb[:, :3], [0.299, 0.587, 0.114])

    # step 1: Reduce noise - Gaussian filter 5x5
    def gaussian_filter(self, img_gray, sigma1=1.4, sigma2=1.4):

        gau_sum = 0
        gaussian = np.zeros([5, 5])
        for i in range(5):
            for j in range(5):
                gaussian[i, j] = np.exp(-1/2 * (np.square(i-3) / np.square(sigma1)
                        + (np.square(j-3) / np.square(sigma2)))) / 2 * math.pi * sigma1 * sigma2
                gau_sum += gaussian[i, j]

        gaussian = gaussian / gau_sum

        W, H = img_gary.shape
        new_gary = np.zeros([W-5, H-5])

        for i in range(W-5):
            for j in range(H-5):
                new_gary[i, j] = np.sum(img_gary[i:i+5, j:j+5] * gaussian)

        return new_gary

    # step 2: Calculate gradient
    def gradients(self, new_gary):

        W1, H1 = new_gary.shape
        dx = np.zeros([W1-1, H1-1])
        dy = np.zeros([W1-1, H1-1])
        d = np.zeros([W1-1, H1-1])
        theta = np.zeros([W1-1, H1-1])

        for i in range(W1-1):
            for j in range(H1-1):
                dx[i, j] = new_gary[i, j+1] - new_gary[i, j]
                dy[i, j] = new_gary[i+1, j] - new_gary[i, j]
                d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
                theta[i, j] = math.atan(dx[i, j] / (dy[i, j] + 0.00000000001))

        return d, theta

    # step 3: Non-maximum suppression
    def non_max_sup(self):
        pass

    # step 4: Double threshold
    def double_threshold(self):
        pass


if __name__ == '__main__':

    canny = Canny()
    img_gary = canny.gray("E:/GitHub/CV_Learning/image-alignment/football.jpg")

