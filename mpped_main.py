import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import openpyxl
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from natsort import natsorted
import argparse
import torch

plt.clf()

import mpped_detect as detlib

#***********************************************************@@@@*******************************************************

if __name__ == '__main__':
    root=sys.path
    print("********************************* mpped_detect_main.py ***********************************")
    ImageFolder = 'MPPED_data/images/test'
    JPGFileList = os.listdir(ImageFolder)
    JPGFileList= natsorted(JPGFileList)
    print('Number of images:',len(JPGFileList))#,'\n',JPGFileList)

    serial=0

    for image_file in JPGFileList:
        img_fname = ImageFolder + '/' + image_file
        TargetName=img_fname.split('/')[-1]
        print('********************** serial:',serial,' **********************')
        print('TargetName:',TargetName)
        img = cv2.imread(img_fname)
        print("Image loading completed. The following data will be processed:",img_fname)
        IMG_size = img.shape
        print('Shape of image', TargetName, "is:", IMG_size)
        print("Start calling the yolov7 mpped detection function")
########################################################################################################################
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='runs/train/yolov7_mpped_det/weights/best.pt', help='model.pt path(s)')#epoch_024.pt to 29
        parser.add_argument('--source', type=str, default=img_fname ,help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_false', help='display results') #'store_false' to display
        parser.add_argument('--save-txt', action='store_false', help='save results to *.txt') #'store_true' for not saving
        parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_false', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='MPPED_data/result/mpped_detect_crops_&_features', help='save results to project/name')
        parser.add_argument('--name', default='teeth_01_detect', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        opt = parser.parse_args()

        ToyBox, ToyLabel, ToyScore =detlib.run_mpped_detect(opt)


        print('ToyBox shape:', ToyBox.shape, 'ToyLabel shape:', ToyLabel.shape, 'ToyScore shape:',ToyScore.shape)
        print("Toybox...\n", ToyBox)
        print(" ToyLabel...\n", ToyLabel)
        print("ToyScore...\n", ToyScore)

        serial=serial+1


