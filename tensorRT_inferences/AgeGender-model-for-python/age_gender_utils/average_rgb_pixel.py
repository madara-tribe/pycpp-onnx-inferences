import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

def average_rgb_pixcel(img):
    h, w, c = img.shape #height, width, channnel
    l=0
    b_ave=0; g_ave=0; r_ave=0

    for i in range(h):
        for j in range(w):
            #画素値[0,0,0]（Black）を除外してピクセルの和とbgrの画素値の合計を計算する
            if(img[i,j,0] != 0 or img[i,j,1] != 0 or img[i,j,2] != 0 ):
                l+=1    #対象となるピクセル数を計算する
                #対象となるピクセルの画素値の和を計算する
                b_ave=b_ave+img[i,j,0]
                g_ave=g_ave+img[i,j,1]
                r_ave=r_ave+img[i,j,2]

    #画素値合計をピクセル数で除することでRGBの画素値の平均値を求める
    b_ave=b_ave/l
    g_ave=g_ave/l
    r_ave=r_ave/l
    return (b_ave+g_ave+r_ave)/3


def main():
    dirs = '../../UTKface_dataset/UTKFace/*.jpg'
    #dirs = 'UTKface_dataset/UTKFace/25_1_2_20170104021412148.jpg.chip.jpg'
    imgs = np.array([cv2.imread(img) for idx, img in enumerate(glob.glob(dirs)) if idx<100])
    avg = [average_rgb_pixcel(img) for img in imgs]
    print(np.array(avg).sum()/len(imgs))
    
if __name__=='__main__':
    main()
