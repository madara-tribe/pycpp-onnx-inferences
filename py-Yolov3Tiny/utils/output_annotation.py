import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import glob
import xml.etree.ElementTree as ET


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--xml_path", type=str, default='VOCdevkit/VOC2012/Annotations/*.xml',
                        help="VOC2012 jpg path")
    parser.add_argument("--jpg_path", type=str, default='VOCdevkit/VOC2012/JPEGImages/*.jpg',
                        help='VOC2012 xml path')
    parser.add_argument("--file_txt", type=str, default='anchors_classes/yolov3_voc.txt',
                        help='annotation text file')
    parser.add_argument("--class_txt", type=str, default='classes.txt',
                        help='VOC2012 xml path')
    return parser


def detect_annotstion(jpgs, xmls, classes_dicts, plot=True):
    img = cv2.imread(jpgs)

    file = open(xmls)
    tree = ET.parse(file)
    root = tree.getroot()

    all_list = None
    img_file = root.find('filename').text  # 画像ファイル名を取得
    x1, y1, x2, y2 = 0, 0, 0, 0
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes_dicts.keys() or int(difficult)==1:
            continue
        xmlbox = obj.find('bndbox')
        try:
            x1, y1, x2, y2 = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        except ValueError:
            pass
        if plot:
            bbox = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=10)
            plt.imshow(bbox),plt.show()
        #print(b, cls, img_file)
        cls_idx = classes_dicts[cls]
        #print(cls, cls_idx)
        yield [img_file, x1, y1, x2, y2, cls_idx]

def create_anno_txt(txt_path='anchors_classes/yolov3_voc.txt'):
    classes = ['car', 'bus', 'bicycle', 'boat', 'motorbike', 'train']
    classes_dicts = {key:idx for idx, key in enumerate(classes)}
    opts = get_argparser().parse_args()
    xmls = sorted(glob.glob(opts.xml_path))
    jpgs = sorted(glob.glob(opts.jpg_path))
    df=[]
    for jpg, xml in zip(jpgs, xmls):
        df += detect_annotstion(jpg, xml, classes_dicts, plot=False)
    with open(opts.file_txt, 'w+') as f:
        for d in df:
            f.write(','.join(map(str, d)) + '\n')
            
    print('class_txt lenth', len(df))
    with open(opts.class_txt, 'wt') as f:
        for ele in classes:
            f.write(ele+'\n')
    
def open_anno_txt(txt_file='anchors_classes/yolov3_voc.txt'):
    with open(txt_file) as f:
        class_names = f.readlines()
    return [c.strip() for c in class_names]


if __name__=='__main__':
    create_anno_txt()
    #xml_path= 'VOC2012/Annotations/*.xml'
    #jpg_path = 'VOC2012/JPEGImages/*.jpg' 
