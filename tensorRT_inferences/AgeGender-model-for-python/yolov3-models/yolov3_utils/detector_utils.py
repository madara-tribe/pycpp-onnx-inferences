import os, sys
import numpy as np
import cv2

from .load import to_mean_pixel, MEAN_AVG
from logging import getLogger
logger = getLogger(__name__)

minus = -2
add = 2

def preprocessing_img(img):
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


def load_image(image_path):
    if os.path.isfile(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        logger.error(f'{image_path} not found.')
        sys.exit()
    return preprocessing_img(img)


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def plot_results(detector, img, category, segm_masks=None, logging=True):
    """
    :param detector: ailia.Detector, or list of ailia.DetectorObject
    :param img: ndarray data of image
    :param category: list of category_name
    :param segm_masks:
    :param logging: output log flg
    :return:
    """
    h, w = img.shape[0], img.shape[1]
    count = detector.get_object_count() if hasattr(detector, 'get_object_count') else len(detector)
    if logging:
        print(f'object_count={count}')

    # prepare color data
    colors = []
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        print('allabj', obj)
        # print result
        if logging:
            print(f'+ idx={idx}')
            print(
                f'  category={obj.category}[ {category[obj.category]} ]'
            )
            print(f'  prob={obj.prob}')
            print(f'  x={obj.x}')
            print(f'  y={obj.y}')
            print(f'  w={obj.w}')
            print(f'  h={obj.h}')

        color = hsv_to_rgb(256 * obj.category / (len(category) + 1), 255, 255)
        colors.append(color)

    # draw segmentation area
    if segm_masks:
        for idx in range(count):
            mask = np.repeat(np.expand_dims(segm_masks[idx], 2), 3, 2).astype(np.bool)
            color = colors[idx][:3]
            fill = np.repeat(np.repeat([[color]], img.shape[0], 0), img.shape[1], 1)
            img[:, :, :3][mask] = img[:, :, :3][mask] * 0.7 + fill[mask] * 0.3

    print('age gender prediction point start')
    # draw bounding box
    img_size = 299
    
    crop_faces = []
    crop_pos = []
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        
        # face crop
        lmin = int(h * obj.y)
        lmax = int(w * (obj.x + obj.w))+add
        rmin = int(w * obj.x)+minus
        rmax = int(h * (obj.y + obj.h))+add
        
        cimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGRA2BGR)
        
        top_left = (int(w * obj.x)+minus, int(h * obj.y)+minus)
        bottom_right = (int(w * (obj.x + obj.w))+add, int(h * (obj.y + obj.h))+add)
        
        color = colors[idx]
        
        # image normalize for age-gender
        crop_face_resize = cv2.resize(cimg[lmin:rmax,rmin:lmax], (img_size, img_size))
        crop_face_resize = to_mean_pixel(crop_face_resize, MEAN_AVG)
        crop_faces.append(crop_face_resize.astype(np.float32)/255)
        crop_pos.append([rmin, rmax])
        
        cv2.rectangle(img, top_left, bottom_right, color, 4)

    # draw label
    for idx in range(count):
        obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
        fontScale = w / 2048

        text = category[obj.category] + " " + str(int(obj.prob*100)/100)
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)[0]
        tw = textsize[0]
        th = textsize[1]

        margin = 3

        top_left = (int(w * obj.x), int(h * obj.y))
        bottom_right = (int(w * obj.x) + tw + margin, int(h * obj.y) + th + margin)
        
        color = colors[idx]
        cv2.rectangle(img, top_left, bottom_right, color, thickness=-1)

        text_color = (255,255,255,255)
        cv2.putText(img, text, (top_left[0], top_left[1] + th),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale, text_color, 1)
    print('age gender prediction point endpoint')
    return img, np.array(crop_faces), crop_pos

