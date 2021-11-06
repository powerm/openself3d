import torch 
from torch.utils.data import dataset 

import os
import copy 
import numpy as np 
import random 
import glob
import sys 


class DenseCorrDataSource(object):
    
    def __init__(self):
        pass 
    
    
    def scene_generator(self):
        """
        Returns an generator that traverses all the scenes
        :return:
        :rtype:
        """
        NotImplementedError("subclass must implement this method")
        
    
    def init_length(self):
        """
        Computes the total number of images and scenes in this dataset 
        Sets the resualt to class variable self.num_images_total and self._num_scenes
        """
        raise NotImplementedError("subclass must implement this method")

    
    def get_random_scene_name(self):
        """return a random scene_name
        """
        raise NotImplementedError("subclass must implement this method")
    
    def get_full_path_for_scene(self, scene_name):
        raise NotImplementedError("subclass must implement this method")
    
    def get_random_image_index(self, scene_name):
        """return a random image index form a given scene

        Args:
            scene_name ([str]): [description]
        
        """
        raise NotImplementedError("subclass must implement this method")
    
    def get_image_filename(self, scene_name, img_index, image_type):
        raise NotImplementedError("Implement in superclass")
    
    def get_pose_list(self, scene_name):
        raise NotImplementedError("subclass must implement this method")
    
    def get_pose_from_scene_name_and_idx(self, scene_name, idx):
        """[summary]

        Args:
            scene_name ([str]): [description]
            idx ([int]): [description]
        return 4x4 numpy array
        """
        raise NotImplementedError("subclass must implement this method")
    

    

        
        