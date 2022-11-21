import os
import re
import cv2
import ast
import random
import pandas as pd
import numpy as np

from PIL import Image
import albumentations as A
from collections import namedtuple
from albumentations.pytorch.transforms import ToTensorV2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

from utils import *


def draw_rect(img, bboxes, color=(255, 0, 0)):
    img = img.copy()
    for bbox in bboxes:
        bbox = np.array(bbox).astype(int)
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        img = cv2.rectangle(img, pt1, pt2, color, int(max(img.shape[:2]) / 200))
    return img

def plot_multiple_img(img_matrix_list, title_list, ncols, nrows=3, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
        myaxes[i // ncols][i % ncols].grid(False)
        myaxes[i // ncols][i % ncols].set_xticks([])
        myaxes[i // ncols][i % ncols].set_yticks([])

    plt.show()

pathxml = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/annotations/*.xml"
pathImages = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/images"

df_train = generateDfFromXML(pathxml, pathImages)

class_dict = {'speedlimit':0, 'stop':1, 'crosswalk':2, 'trafficlight':3}

df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])
df_train = add_bbox_coordinate(df_train)
df_train = add_bbox_per_pixel(df_train)

# print(df_train.head())

chosen_img = cv2.imread(df_train.iloc[0][0])
bboxes = [[df_train.iloc[0][4],df_train.iloc[0][5],df_train.iloc[0][6],df_train.iloc[0][7]]]

bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']}

albumentation_list = [A.Compose([A.RandomFog(p=1)], bbox_params=bbox_params),
                      A.Compose([A.RandomBrightness(p=1)], bbox_params=bbox_params),
                      A.Compose([A.RandomCrop(p=1, height=256, width=256)], bbox_params=bbox_params), 
                      A.Compose([A.Rotate(p=1, limit=90)], bbox_params=bbox_params),
                      A.Compose([A.RGBShift(p=1)], bbox_params=bbox_params), 
                      A.Compose([A.HorizontalFlip(p=1)], bbox_params=bbox_params),
                      A.Compose([A.VerticalFlip(p=1)], bbox_params=bbox_params), 
                      A.Compose([A.RandomContrast(limit=0.5, p = 1)], bbox_params=bbox_params)
                     ]

titles_list = ["Original", 
               "RandomFog",
               "RandomBrightness", 
               "RandomCrop",
               "Rotate", 
               "RGBShift", 
               "HorizontalFlip", 
               "VerticalFlip", 
               "RandomContrast"]


img_matrix_list = [draw_rect(chosen_img, bboxes)]

for aug_type in albumentation_list:
    anno = aug_type(image=chosen_img, bboxes=bboxes, labels=np.ones(len(bboxes)))
    print(anno['image'].shape, np.max(anno['image']),np.min(anno['image']),anno['bboxes'])
    # img  = draw_rect(anno['image'], anno['bboxes'])
    # img_matrix_list.append(img)
    
# plot_multiple_img(img_matrix_list, 
#                   titles_list, 
#                   ncols = 3, 
#                   main_title="Different Types of Augmentations")
    
