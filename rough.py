#library imports
import os
import random
import math
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np

import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import glob


# path_images = "/media/sharat/New Volume1/radhesh/bboxPrediction/data/images/*.png"
# path_annotations = "/media/sharat/New Volume1/radhesh/bboxPrediction/data/annotations/*.xml"

# images = glob.glob(path_images)
# annotations = glob.glob(path_annotations)
# print(len(images), len(annotations))

def generateDfFromXML(path_xml, path_img):
    
    annotations = glob.glob(path_xml)
    anno_list = []

    for file in annotations:
        root = ET.parse(file).getroot()
        
        anno = {}
        
        anno['filename'] = path_img + "/" + root.find("./filename").text
        anno['width'] = root.find("./size/width").text
        anno['height'] = root.find("./size/height").text
        anno['class'] = root.find("./object/name").text
        anno['xmin'] = int(root.find("./object/bndbox/xmin").text)
        anno['ymin'] = int(root.find("./object/bndbox/ymin").text)
        anno['xmax'] = int(root.find("./object/bndbox/xmax").text)
        anno['ymax'] = int(root.find("./object/bndbox/ymax").text)
        
        anno_list.append(anno)

    return pd.DataFrame(anno_list)

df_train = generateDfFromXML("/media/sharat/New Volume1/radhesh/bboxPrediction/data/annotations/*.xml",
                             "/media/sharat/New Volume1/radhesh/bboxPrediction/data/images")


class_dict = {'speedlimit':0, 'stop':1, 'crosswalk':2, 'trafficlight':3}

df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])

# print(df_train.head())


def create_mask(bb, x):
    rows, cols, *_ = x.shape
    Y = np.zeros((rows, cols))
    Y[int(bb[0]):int(bb[2]), int(bb[1]):int(bb[3])] = 1
    return Y


def mask_to_bb(Y):

    cols, rows = np.nonzero(Y)
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x):
    return np.array([x[5],x[4],x[7],x[6]])

def resize_image_bb(read_path,bb,sz):
    im = cv2.cvtColor(cv2.imread(read_path), cv2.COLOR_BGR2RGB)
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.49*sz), sz))
    return mask_to_bb(Y_resized)
    
new_bbs = []
for index, row in df_train.iterrows():
    new_bb = resize_image_bb(row['filename'], create_bb_array(row.values),300)
    new_bbs.append(new_bb)
    
df_train['new_bb'] = new_bbs

# print(df_train.head())

# img = cv2.cvtColor(cv2.imread(df_train['filename'].loc[0]), cv2.COLOR_BGR2RGB)
# bbox = create_bb_array(df_train.iloc[0].values)
# Y = create_mask(bbox, img)
# print(img.shape)
# print(Y.shape)
# print('bbox initial: ', bbox)

# img = cv2.resize(img, (400,300))
# Y = cv2.resize(Y, (400,300))
# print(img.shape)
# print(Y.shape)
# bbox = mask_to_bb(Y)
# print('bbox final: ', bbox)


# plt.figure()
# plt.imshow(img)
# plt.show()
# bbox = [df_train['ymin'].loc[0], df_train['xmin'].loc[0], df_train['ymax'].loc[0], df_train['xmax'].loc[0]]
# print(bbox)
# mask = create_mask(bbox, img)

# plt.figure()
# plt.imshow(mask)
# plt.show()

def resizeImg(read_path, sz):
    img = cv2.cvtColor(cv2.imread(read_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(1.49*sz), sz))
    return img

def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]

# random crop to the original size
def random_crop(x, r_pix=8):

    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):

    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):

    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    # x = cv2.imread(str(path)).astype(np.float32)
    # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    x = resizeImg(path, 300).astype(np.float32)
    x = x/255.0
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5: 
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)

def normalize(im):
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
    plt.show()
    

# im = resizeImg(df_train.iloc[68].values[0], 300)
# show_corner_bb(im, df_train.iloc[68].values[8])

# im, bb = transformsXY(df_train.iloc[68].values[0], df_train.iloc[68].values[8], True)
# show_corner_bb(im, bb)

df_train = df_train.reset_index()
X = df_train[['filename', 'new_bb']]
Y = df_train['class']

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

class RoadDataset(Dataset):
    def __init__(self, paths, bb, y, transforms=False):
        self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        x, y_bb = transformsXY(path, self.bb[idx], self.transforms)
        x = normalize(x)
        print('inside roaddataset: ',x.shape)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb
    
train_ds = RoadDataset(X_train['filename'],X_train['new_bb'] ,y_train, transforms=True)
valid_ds = RoadDataset(X_val['filename'],X_val['new_bb'],y_val)

batch_size = 8
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), self.bb(x)
    

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr
        

best_val_acc = 0.0

def train_epocs(model, optimizer, train_dl, val_dl, epochs=10,C=1000):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_class, y_bb in train_dl:
            batch = y_class.shape[0]
            x = x.cuda().float()
            y_class = y_class.cuda()
            y_bb = y_bb.cuda().float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        val_loss, val_acc = val_metrics(model, valid_dl, C)
        print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))
        
        # if val_acc > best_val_acc:
        #     print('val acc impored, saving weights...')
        #     torch.save(model, "/media/sharat/New Volume1/radhesh/bboxPrediction/output/bestweight.pth")
        #     best_val_acc = val_acc
        # else:
        #     print('Val Acc didnot improve.')
        torch.save(model, "/media/sharat/New Volume1/radhesh/bboxPrediction/output/bestweight.pth")
        
    return sum_loss/total


def val_metrics(model, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    for x, y_class, y_bb in valid_dl:
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda()
        y_bb = y_bb.cuda().float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb/C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    
    return sum_loss/total, correct/total


model = BB_model().cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.006)

train_epocs(model, optimizer, train_dl, valid_dl, epochs=15)

update_optimizer(optimizer, 0.001)

train_epocs(model, optimizer, train_dl, valid_dl, epochs=10)

model = torch.load("/media/sharat/New Volume1/radhesh/bboxPrediction/output/bestweight.pth")
model = model.cuda()

im = resizeImg('/media/sharat/New Volume1/radhesh/bboxPrediction/data/images/road82.png', 300)
cv2.imwrite('/media/sharat/New Volume1/radhesh/bboxPrediction/output/road82.jpg',im)

test_ds = RoadDataset(pd.DataFrame([{'path':'/media/sharat/New Volume1/radhesh/bboxPrediction/output/road82.jpg'}])['path'],pd.DataFrame([{'bb':np.array([0,0,0,0])}])['bb'],pd.DataFrame([{'y':[0]}])['y'])
x, y_class, y_bb = test_ds[0]

xx = torch.FloatTensor(x[None,])

out_class, out_bb = model(xx.cuda())

print("Predicted class: ",torch.max(out_class, 1))
print("it should have been: ", y_class)

bb_hat = out_bb.detach().cpu().numpy()
bb_hat = bb_hat.astype(int)

# x = np.reshape(x, (x.shape[1], x.shape[2], x.shape[0]))
# plt.imshow(x)
# plt.show()

show_corner_bb(im, bb_hat[0])