import random
import pandas as pd
import numpy as np

import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F

import glob


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


#Load Data From Json file.

# class customDataset(Dataset):
    
#   def __init__(self, json_path, img_path, numclasses, transforms = False):
#     self.data = json.load(open(json_path))
#     self.data_annotations = self.data['annotations']
#     self.img_path = img_path
#     self.numclasses = numclasses
#     self.transforms = transforms

#   def __len__(self):
#     return len(self.data_annotations)

#   def __getitem__(self, idx):
#     img_path = self.img_path + "/" + self.data['images'][self.data_annotations[idx]["image_id"]]['file_name']
#     img = cv2.imread(img_path)
    
#     try:
#       (h,w) = img.shape[:2]
    
#     except: 
#       print(img_path)
#       sys.exit()
#     [xmin,ymin,width,height] = self.data_annotations[idx]['bbox']

#     xmin = float(xmin)/w
#     ymin = float(ymin)/h
#     xmax = float(xmin+width)/w
#     ymax = float(ymin+height)/h

#     bbox = np.array([xmin,ymin,xmax,ymax], dtype = np.float32)

#     label = np.array(self.data_annotations[idx]['category_id'], dtype = np.int)
    
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224,224))

#     if self.transforms:
#       img = self.transforms(img)

#     return img, label, bbox
    





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







def add_bbox_coordinate(df):
    new_bbs = []
    for index, row in df.iterrows():
        new_bb = resize_image_bb(row['filename'], create_bb_array(row.values),300)
        new_bbs.append(new_bb)
        
    df['new_bb'] = new_bbs
    return df    

def add_bbox_per_pixel(df):
    
    new_bbs = []
    for index, row in df.iterrows():
        arr = np.array([round(row[4]/float(row[1]),6), round(row[5]/float(row[2]),6), round(row[6]/float(row[1]),6), round(row[7]/float(row[2]),6)])
        new_bbs.append(arr)
    df['bb_pixel'] = new_bbs
    return df

def add_bbox_co(df):

    new_bbs = []
    for index, row in df.iterrows():
        arr = np.array([row[4],row[5],row[6], row[7]])
        new_bbs.append(arr)
    df['bb'] = new_bbs
    return df










def resizeImg(read_path, sz):
    img = cv2.cvtColor(cv2.imread(read_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(1.49*sz), sz))
    return img







#AUGMENTATIONS

def crop(im, r, c, target_r, target_c): 
    return im[r:r+target_r, c:c+target_c]


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



# PLOT BBOX ON IMAGE

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)



def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))
    plt.show()
    





#MISC

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr


        
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

def plotGraphandSave(H, savePath, acc = True):
    
    plt.figure()
    
    if acc:
        plt.plot(H["total_train_acc"], label = "train_acc")
        plt.plot(H["total_val_acc"], label = "val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
    
    else:
            
        plt.plot(H["total_train_loss"], label = "train_loss")
        plt.plot(H["total_val_loss"], label = "val_loss")
        plt.plot(H["total_train_bbox_loss"], label = "train_bbox_loss")
        plt.plot(H["total_train_class_loss"], label = "train_class_loss")
        plt.plot(H["total_val_bbox_loss"], label = "val_bbox_loss")
        plt.plot(H["total_val_class_loss"], label = "val_class_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
    
    plt.legend(loc = "upper right")
    plt.savefig(savePath)
    
def iou(bb1, bb2):
    
    y1 = max(bb1[0], bb2[0])
    x1 = max(bb1[1], bb2[1])
    y2 = min(bb1[2], bb2[2])
    x2 = min(bb1[3], bb2[3])
    intersect_area = (x2 - x1)*(y2 - y1)

    bb1_area = (bb1[3] - bb1[1])*(bb1[2]- bb1[0])
    bb2_area = (bb2[3] - bb2[1])*(bb2[2]- bb2[0])
    union_area = (bb1_area + bb2_area) - intersect_area

    # print('intersect area: ', intersect_area)
    # print('bb1 area: ', bb1_area)
    # print('bb2 area: ', bb2_area)
    
    iou = intersect_area / union_area

    return iou

def convert_bbox_format(bb):
    return np.array([bb[1], bb[0], bb[3], bb[2]])