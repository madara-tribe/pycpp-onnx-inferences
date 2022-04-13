import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys, os
import glob

def mask_to_indexmap(masks, plot=False):
    colormap=[]
    for idx, mask in enumerate(masks):
        if len(np.shape(mask))==3:
            mask_h, mask_w, _ = np.shape(mask)
        else:
            mask_h, mask_w = np.shape(mask)
        masked = np.zeros([mask_h, mask_w, 3], dtype=np.uint8)
        for h in range(mask_h):
            for w in range(mask_w):
                class_id = mask[h, w]
                #print(idx, np.unique(class_id))
                r, b, g = (0, 0, 0)
                if class_id == 0:
                    r, g, b = (0, 0, 0) # black
                elif class_id == 1:
                    r, g, b = (255, 0, 0) # black
                elif class_id == 2:
                    r, g, b = (255,255,0) # green
                elif class_id == 3:
                    r, g, b = (0,0,255) # green
                elif class_id == 4:
                    r, g, b = (128,0,128) # green
                else:
                    r, g, b = (255,255,255) # white

                masked[h, w, 0] = r
                masked[h, w, 1] = g
                masked[h, w, 2] = b
        #if plot:
            #plt.imshow(masked),plt.show()
        print(idx, masked.shape)
        colormap.append(masked)
    return colormap

if __name__=='__main__':
    train = False
    w, h = 256, 256
    if train:
        dirname = 'data/B/train'
        name = 'train'
        paths = [p for p in glob.glob('../../4cls/anno/*.png')]
        paths.sort()
        img = [cv2.imread(p, 0) for p in paths]
    else:
        dirname = 'data/B/test'
        name='test'
        paths = [p for p in glob.glob('../../4cls/val_anno/*.png')]
        paths.sort()
        img = [cv2.imread(p, 0) for p in paths]
    print('num data is ', len(paths))
    img = [cv2.resize(im, (w, h)) for im in img]
    masks = mask_to_indexmap(img)
    for idx, mask in enumerate(masks):
        mask = mask.astype(np.uint8)
        n = '0'+str(idx)+'.png' if idx<10 else str(idx)+'.png'
        print(os.path.join(dirname, name+n))
        cv2.imwrite(os.path.join(dirname, name+n), mask)
