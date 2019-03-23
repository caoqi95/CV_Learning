import matplotlib.pyplot as plt
import numpy as np

# plot result
def plot_res(img, img_eq, name1, name2):

    # plotting for orignal image        
    hist, bins = np.histogram(img.flatten(), 256)
    pdf = hist / np.prod(img.shape)
    plt.figure(figsize=(10,8))
    plt.subplot(2, 2, 1)
    plt.plot(pdf, color='r')
    plt.title(name1)
    plt.xlim([0, 256])
    plt.ylim([0, pdf.max()+0.005])
    plt.subplot(2, 2, 2)
    plt.imshow(img, cmap="gray")

    # plotting for equalizated image
    hist2, bin2 = np.histogram(img_eq, 256)
    pdf2 = hist2 / np.prod(img.shape)
    plt.subplot(2, 2, 3)
    plt.plot(pdf2, color='r')
    plt.title(name2)
    plt.xlim([0, 256])
    plt.ylim([0, pdf2.max()+0.005])    
    plt.subplot(2, 2, 4)   
    plt.imshow(img_eq, cmap="gray")


# plot hist of a image
def plot_hist(img):
    
    plt.hist(img.flatten(), 256)