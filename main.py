from utils import *
# from dataloader import RoadDataset
from train_dataloader import RoadDataset
# from test_dataloader import RoadDatasetTest
from model import BB_model
from torch.utils.data import DataLoader
from train import train_one_epoc, val_one_epoch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import time
import sys

def main():
    
    pathxml = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/annotations/*.xml"
    pathImages = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/data/images"
    savePath = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/output/bestweight_self1.pth"
    plotSavePathAcc = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/output/acc_curve.png"
    plotSavePathLoss = "/media/sharat/New Volume1/radhesh/bboxPredictionSelf/output/loss_curve.png"
    
    History = {
                "total_train_loss": [], "total_val_loss": [], 
                "total_train_acc": [], "total_val_acc": [], 
                "total_train_bbox_loss":[], "total_val_bbox_loss":[],
                "total_train_class_loss":[], "total_val_class_loss":[]
              }
    
    batch_size = 8
    epochs = 20
    min_val_loss = 1e6
    best_val_acc = 0.0
    lr = 5e-5
    
    df_train = generateDfFromXML(pathxml, pathImages)

    class_dict = {'speedlimit':0, 'stop':1, 'crosswalk':2, 'trafficlight':3}

    df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])
    df_train = add_bbox_coordinate(df_train)
    df_train = add_bbox_per_pixel(df_train)
    
    df_train = df_train.reset_index()
    
    # print(df_train.head())
    
    X = df_train[['filename', 'bb_pixel']]
    Y = df_train['class']
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])
    
    # train_ds = RoadDataset(X_train['filename'],X_train['new_bb'] ,y_train, transforms=True)
    # valid_ds = RoadDataset(X_val['filename'],X_val['new_bb'],y_val)
    
    train_ds = RoadDataset(X_train,y_train, transforms=train_transforms)
    valid_ds = RoadDataset(X_val,y_val, transforms=val_transforms)
    
    # test_ds = RoadDatasetTest(X_val, y_val, False)
    # test_ds = valid_ds
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    
    # test_dl = DataLoader(test_ds, batch_size = batch_size)
    
    model = BB_model().cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    scheduler = StepLR(optimizer,  step_size = 7, gamma = 0.7, verbose = True)
    
    print("Beginning training for {} epochs...".format(epochs))
    startTime = time.time()
    
    for i in range(epochs):
        
        print("epoch:",i+1)
        
        train_loss, train_acc, train_bbox_loss, train_class_loss = train_one_epoc(model, optimizer, train_dl)
        print("train_loss: %.3f train_acc: %.3f train_bbox_loss: %.3f train_class_loss: %.3f" % (train_loss, train_acc, train_bbox_loss, train_class_loss))
        scheduler.step()
        
        val_loss, val_acc, val_bbox_loss, val_class_loss = val_one_epoch(model, valid_dl, C=1000)
        print("val_loss: %.3f val_acc: %.3f val_bbox_loss: %.3f val_class_loss: %.3f" % (val_loss, val_acc, val_bbox_loss, val_class_loss))
        
        if min_val_loss > val_loss:
            print("val loss decreased. Saving model...")
            torch.save(model, savePath)
            min_val_loss = val_loss
            best_val_acc = val_acc
            
            
        History["total_train_loss"].append(train_loss)
        History["total_train_acc"].append(train_acc)
        History["total_val_loss"].append(val_loss)
        History["total_val_acc"].append(val_acc)
        History["total_train_bbox_loss"].append(train_bbox_loss)
        History["total_train_class_loss"].append(train_class_loss)
        History["total_val_bbox_loss"].append(val_bbox_loss)
        History["total_val_class_loss"].append(val_class_loss)
        
    
    print("saving loss / accuracy curve")    
    plotGraphandSave(History, plotSavePathAcc, True)
    plotGraphandSave(History, plotSavePathLoss, False)
        
    print("Best Val Accuracy achieved -- ", best_val_acc)
    print("Minimum Val Loss achieved -- ", min_val_loss)
    
    endTime = time.time()
    print("total time taken to train the model: {:.2f}s".format(endTime - startTime))
    
if __name__ == "__main__":
    main()