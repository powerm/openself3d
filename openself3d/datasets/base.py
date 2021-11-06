from abc import ABC, ABCMeta,abstractmethod

import torch 

from torch.utils.data import Dataset

from torchvision.transforms import Compose

from mmcv import build_from_cfg

from .build import DATASETS, PIPELINES, build_datasource



class BaseDataset(Dataset, metaclass=ABCMeta):
    
    
    def __init__(self, cfg, pipeline):
        self.data_source = build_datasource(cfg)
        pipelines = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipelines)
        
    def __len__(self):
        return self.data_source.get_length()
    
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
    @abstractmethod
    def evaluate(self, scores, keyword, logger=None, **kwargs):
        pass
