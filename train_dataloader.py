from utils import *
import numpy as np
from torch.utils.data import Dataset
import sys

class RoadDataset(Dataset):
    def __init__(self, df, y, transforms=False):
        
        self.df = df
        self.transforms = transforms
        self.y = y
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        path = self.df.iloc[idx][0]
        y_class = self.y.iloc[idx]
        x = resizeImg(path, 300)
        y_bb = self.df.iloc[idx][1]
        
        if self.transforms:
            x = self.transforms(x)
        
        return x, y_class, y_bb