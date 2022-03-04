import torch 
from .build import DATASETS
from .base import BaseDataset
from .data_sources import  SpartanDataSource
from  torch.utils.data  import Dataset
import numpy as np 

def flatten_uv_tensor(uv_tensor, image_width):
    """
        Flattens a uv_tensor to single dimensional tensor
        :param uv_tensor:
        :type uv_tensor:
        :return:
        :rtype:
    """
    
    return uv_tensor[1] * image_width + uv_tensor[0]


@DATASETS.register_module
class ContrastiveDataset(Dataset):
    
    
    def __init__(self, data_source_cfg, train_config, mode="train"):
        super(ContrastiveDataset, self).__init__()
        
        self.dataSource = SpartanDataSource(data_source_cfg, mode= mode)
        self.dataSource.set_parameters_from_training_config(train_config)
    
    
    def __getitem__(self, idx):
        
        data_load_type = self.dataSource._get_data_load_type()
        
        while(True):
            datatype, \
            image_a_rgb, image_b_rgb, \
            image_a_depth_numpy, image_b_depth_numpy,  \
            uv_a, uv_b, masked_non_matches_a, masked_non_matches_b,  \
            background_non_matches_a, background_non_matches_b,  \
            blind_non_matches_a, blind_non_matches_b, metadata=self.dataSource.get_sample()
            
            if datatype != -1:
                image_width = image_a_rgb.shape[1]
                image_height = image_a_rgb.shape[0]
                uv_a = flatten_uv_tensor(uv_a, image_width)
                uv_b = flatten_uv_tensor(uv_b, image_width)
                masked_non_matches_a = flatten_uv_tensor(masked_non_matches_a, image_width)
                masked_non_matches_b = flatten_uv_tensor(masked_non_matches_b, image_width)
                background_non_matches_a = flatten_uv_tensor(background_non_matches_a, image_width)
                background_non_matches_b = flatten_uv_tensor(background_non_matches_b, image_width)
                blind_non_matches_a = flatten_uv_tensor(blind_non_matches_a, image_width)
                blind_non_matches_b = flatten_uv_tensor(blind_non_matches_b, image_width)
                return    datatype, image_a_rgb, image_b_rgb, \
                                                     np.expand_dims(image_a_depth_numpy,axis=0), np.expand_dims(image_a_depth_numpy,axis=0),  \
                                                     uv_a, uv_b, masked_non_matches_a, masked_non_matches_b,  \
                                                     background_non_matches_a, background_non_matches_b,  \
                                                     blind_non_matches_a, blind_non_matches_b, metadata
    
    
    def __len__(self):
        
        return self.dataSource.num_images_total
        