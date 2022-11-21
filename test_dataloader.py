from utils import *
import numpy as np
from torch.utils.data import Dataset
import sys

class RoadDatasetTest(Dataset):
    def __init__(self, df, transforms=False):
        
        self.df = df
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        path = self.df.iloc[idx][0]
        # print(y_class)
        x = resizeImg(path, 300)
        
        if self.transforms:
            x = self.transforms(x)
        
        return x, path