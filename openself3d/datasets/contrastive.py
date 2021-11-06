import torch 
from .build import DATASETS
from .base import BaseDataset





@DATASETS.register_module
class ContrastiveDataset(BaseDataset):
    
    
    def __init__(self, data_source_cfg, pipeline):
        super(ContrastiveDataset, self).__init__(data_source_cfg, pipeline)
    
    
    def __getitem__(self, idx):
        