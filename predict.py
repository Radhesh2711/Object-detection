#library imports
import pandas as pd
import numpy as np
from utils import *
from dataloader import RoadDataset
from tqdm.auto import tqdm
import sys
from torchvision import transforms
import torch.nn as nn

def show_corner_bb_with_label(im, bb, label):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect_1(bb))
    plt.text(bb[0], bb[1], label, fontsize = 15, weight='bold')
    plt.show()
    
    
def create_corner_rect_1(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], color=color,
                         fill=False, lw=3)
    

def predictAndPlotOneImage(imgpath, model):
    
    class_dict = {0:'speedlimit', 1:'stop', 2:'crosswalk', 3:'trafficlight'}
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    img = resizeImg(imgpath, 300)
    x = test_transforms(img)
    
    x = torch.FloatTensor(x[None,])
    
    x = x.cuda().float()
    out_class, out_bb = model(x)
    print('out_class: ',out_class)
    print("Predicted Class: ", class_dict[torch.argmax(out_class, 1).item()])
    print("out_bb: ", out_bb)
    
    softmax_outClass = nn.Softmax(dim=1)(out_class)
    print('softmax_outclass: ',softmax_outClass)
    confidenceScore = torch.max(softmax_outClass).item()
    
    bb_pred  = out_bb.detach().cpu().numpy().astype(int)
    
    show_corner_bb_with_label(img, bb_pred[0], class_dict[torch.argmax(out_class, 1).item()] + ": " + str(round(confidenceScore, 3)))
    
model = torch.load("/media/sharat/New Volume1/radhesh/bboxPredictionSelf/output/bestweight_self_lr5e-5.pth")
model = model.cuda()
model.eval()

predictAndPlotOneImage("/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/images/road61.png", model)

# pathxml = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/annotations/*.xml"
# pathImages = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/images"

# df_train = generateDfFromXML(pathxml, pathImages)

# class_dict = {'speedlimit':0, 'stop':1, 'crosswalk':2, 'trafficlight':3}

# df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])
# df_train = add_bbox_coordinate(df_train)
# df_train = add_bbox_per_pixel(df_train)

# print(df_train.head())

# im = cv2.cvtColor(cv2.imread(df_train.iloc[0][1]), cv2.COLOR_BGR2RGB)
# show_corner_bb_with_label(im, [df_train.iloc[0][5], df_train.iloc[0][6], df_train.iloc[0][7], df_train.iloc[0][8]], "true" + ": " + str(100))