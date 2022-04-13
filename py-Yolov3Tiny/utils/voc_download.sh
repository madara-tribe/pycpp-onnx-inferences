#!/bin/sh
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
python3 output_annotation.py
mv yolov3_voc.txt classes.txt anchors_classes/

