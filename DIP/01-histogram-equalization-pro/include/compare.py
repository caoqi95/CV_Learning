import matplotlib.pyplot as plt
import numpy as np
import cv2
import time 

from include import enhanceContrast
from include.pplot import plot_res, plot_hist

def compare(img_path):
    
    img = cv2.imread(img_path, 0)
    cv2.imwrite('./result/original.jpg', img)
    enhance_Contrast = enhanceContrast.enhanceContrast()
    img_he = enhance_Contrast.basic_HE(img)
    cv2.imwrite('./result/he.jpg', img_he)
    img_bhe = enhance_Contrast.BHE(img)
    cv2.imwrite('./result/bhe.jpg', img_bhe)
    plot_hist(img)
    plt.show()
    
    limit = int(input("Please input the limit: "))
    img_che = enhance_Contrast.clipped_HE(img, limit)
    cv2.imwrite('./result/che.jpg', img_bhe)
    img_bhepl = enhance_Contrast.BHEPL(img)
    cv2.imwrite('./result/bhepl.jpg', img_bhepl)
    img_ahe = enhance_Contrast.adaptive_HE(img)
    cv2.imwrite('./result/ahe.jpg', img_ahe)
    plot_res(img, img_he, "Orignal", "Basic_HE")
    plot_res(img_bhe, img_che, "Bi-Histogram Equalization", "Clipped Histogram Equalization")
    plot_res(img_bhepl, img_ahe, "Bi-Histogram Equalization with a Plateau Limit", "Adaptive Histogram Equalization")
    