
import mmcv
import numpy as np
from abc import ABCMeta, abstractmethod

from  torch.utils.data import  Dataset



class customRGBD3D(Dataset):
    
    def __init__(name,base_dir):
        self._name = name
        self.base_dir = base_dir
        super.__init__()
        
        
        
    
    def __getitem__(self, index: int) -> T_co:
        return super().__getitem__(index)
    
    def __len__(self) -> int:
        return super().__len__()