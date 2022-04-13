import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


MEAN_AVG = float(130.509485819935)
def to_mean_pixel(img, avg):
    return (img - 128)*(128/avg)

def equalizeHist(img):
    for j in range(3):
        img[:, :, j] = cv2.equalizeHist(img[:, :, j])
    return img

def np_hist_plot(image):
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()
    
    
def plots(imgs, pattern='minmax'):
    assert pattern in ['minmax', 'avg_pixel', 'equalizeHist'], 'Enter correct word [minmax] or [avg_pixel] or [equalizeHist] '
    assert len(imgs)>1, 'imgs must be image not dirpath'
    if pattern=='minmax':
        imgs = np.array(imgs, dtype="float32")/255
        print(imgs.max(), imgs.min())
        np_hist_plot(imgs)
        plt.imshow(imgs[1]),plt.show()
    elif pattern=='avg_pixel':
        imgs = [to_mean_pixel(img, avg=MEAN_AVG) for img in imgs]
        imgs = np.array(imgs, dtype="float32")/255
        print(imgs.max(), imgs.min())
        np_hist_plot(imgs)
        plt.imshow(imgs[1]),plt.show()
    elif pattern=='equalizeHist':
        imgs = [equalizeHist(img) for img in imgs]
        imgs = np.array(imgs, dtype="float32")/255
        print(imgs.max(), imgs.min())
        np_hist_plot(imgs)
        plt.imshow(imgs[1]),plt.show()
        
        
def main():
    dirs = 'UTKface_dataset/UTKFace/*.jpg'
    imgs = [cv2.imread(img) for idx, img in enumerate(glob.glob(dirs)) if idx<100]
    plots(imgs, pattern='minmax')
    plots(imgs, pattern='avg_pixel')
    plots(imgs, pattern='equalizeHist')
    
if __name__=='__main__':
    main()
