import cv2, os
import numpy as np

MEAN_AVG = float(130.509485819935)

def to_mean_pixel(img, avg):
    return (img - 128)*(128/avg)
