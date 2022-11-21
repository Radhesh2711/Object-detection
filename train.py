import torch
import torch.nn.functional as F
from utils import *
from tqdm.auto import tqdm
import sys
from torchvision.ops import complete_box_iou_loss

# import torchvision.ops.complete_box_iou_loss as ciou_loss

def train_one_epoc(model, optimizer, train_dl, C=1000):
    
    model.train()
    
    total = 0
    sum_loss = 0
    correct = 0
    sum_bbox_loss = 0
    sum_class_loss = 0
    
    for (x, y_class, y_bb) in tqdm(train_dl):
        
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda()
        y_bb = y_bb.cuda().float()
        
        out_class, out_bb = model(x)
        _, pred = torch.max(out_class, 1)
        
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        # loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        # loss_bb = F.mse_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = complete_box_iou_loss(out_bb, y_bb, reduction = "sum")
        # loss_bb = loss_bb.sum()
        # loss = loss_class + loss_bb/C
        loss = loss_class + loss_bb
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += batch
        
        sum_loss += loss.item()
        # sum_bbox_loss = sum_bbox_loss + (loss_bb/C).item()
        sum_bbox_loss = sum_bbox_loss + loss_bb.item()
        sum_class_loss = sum_class_loss + loss_class.item()
        
        correct += pred.eq(y_class).sum().item()
    
    train_loss = sum_loss/total
    train_acc = correct/total
    # train_bbox_loss = sum_bbox_loss/total
    # train_class_loss = sum_class_loss/total
    
    return train_loss, train_acc, sum_bbox_loss/total, sum_class_loss/total

def val_one_epoch(model, valid_dl, C=1000):
    
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    sum_bbox_loss = 0
    sum_class_loss = 0
    
    for (x, y_class, y_bb) in tqdm(valid_dl):
        
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda()
        y_bb = y_bb.cuda().float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        # loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        # loss_bb = F.mse_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = complete_box_iou_loss(out_bb, y_bb, reduction = "sum")
        # loss_bb = loss_bb.sum()
        # loss = loss_class + loss_bb/C
        loss = loss_class + loss_bb
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        
        sum_loss += loss.item()
        
        # sum_bbox_loss = sum_bbox_loss + (loss_bb/C).item()
        sum_bbox_loss = sum_bbox_loss + (loss_bb).item()
        sum_class_loss = sum_class_loss + loss_class.item()
        
        total += batch
    
    return sum_loss/total, correct/total, sum_bbox_loss/total, sum_class_loss/total