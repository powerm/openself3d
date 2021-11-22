import os
import copy 
import math
from typing import Counter 
import numpy as np 
import mmcv 
import torch
import glob
import random

from .denseCorr import DenseCorrDataSource


class ImageType:
    RGB = 0
    DEPTH = 1
    MASK = 2
    POINTCLOUD = 4


class SpartanDatasetDataType:
    SINGLE_OBJECT_WITHIN_SCENE = 0
    SINGLE_OBJECT_ACROSS_SCENE = 1
    DIFFERENT_OBJECT = 2
    MULTI_OBJECT = 3
    SYNTHETIC_MULTI_OBJECT = 4


class SceneStructure(object):
    
    def __init__(self, scenes_dir):
        self._scenes_dir = scenes_dir
    
    def fusion_reconstruction_file(self,scene_name):
        
        return os.path.join(self._scenes_dir, scene_name, 'processed','fusion_mesh.ply')
    
    def foreground_fusion_reconstruction_file(self,scene_name):
        
        return os.path.join(self._scenes_dir, scene_name,'processed', 'fusion_mesh_foreground.ply')
    
    def camera_info_file(self,scene_name):
        
        return os.path.join(self._scenes_dir, scene_name,'processed','images', 'camera_info.yaml')
    
    def camera_pose_file(self,scene_name):
        
        return os.path.join(self._scenes_dir,scene_name,'processed','images', 'pose_data.yaml')
    
    def rendered_images_dir(self, scene_name):
        
        return os.path.join(self._scenes_dir, scene_name,'processed', 'rendered_images')
    
    def images_dir(self,scene_name):
        
        return os.path.join(self._scenes_dir, scene_name,'processed','images')
    
    def mask_images_dir(self, scene_name):
        return os.path.join(self._scenes_dir, scene_name, 'processed','image_masks')
    
    def get_image_filename(self, scene_name, img_idx, image_type):
        
        if image_type == ImageType.RGB:
            images_dir = self.images_dir(scene_name)
            file_extension = "_rgb.png"
        elif image_type == ImageType.DEPTH:
            images_dir = self.rendered_images_dir(scene_name)
            file_extension = "_depth.png"
        elif image_type == ImageType.MASK:
            images_dir = self.mask_images_dir(scene_name)
            file_extension = "_mask.png"
        else:
            raise ValueError("unsupported image type")
        
        if isinstance(img_idx, int):
            img_index = utils.getPaddedString(img_index, width = SpartanDataset.PADDED_STRING_WIDTH)
        
        return os.path.join(images_dir, img_index + file_extension)
    
    def get_camera_intrinsics(self, scene_name):
        
        return CameraIntrinsics.from_yaml_file(self.camera_info_file(scene_name))
          
    def get_pose_data(self,scene_name):
        """checks if hvae not aleardy loaded the pose_data.yaml for this scene,
        if haven't then loads it. Then returns the dict of the pose_data.yaml.
        

        Args:
            scene_name ([str]): [description]
        """
        return  mmcv.load(self.camera_pose_file(scene_name))
    
    def get_pose_from_scene_name_and_idx(self,scene_name, idx):
        """return the 4x4 numpy array pose

        Args:
            scene_name ([str]): [description]
            idx ([int]): [description]
        """
        idx = int(idx)
        scene_pose_data = self.get_pose_data(scene_name)
        pose_data = scene_pose_data[idx]['camera_to_world']
        # the function need add
        return utils.homogenous_transform_from_dict(pose_data)
                      
    
    def get_rgbd_mask_pose(self, scene_name, img_idx):
        """Returns rgb image, depth image, mask and pose.

        Args:
            scene_name ([str]): [description]
            img_idx ([int]): [description]
        :return: rgb,depth, mask, pose
        :rtype: PIL.Image.Image, PIL.Image.Image, PIL.Image.IMage, a 4x4 numpy array
        """
        rgb_file = self.get_image_filename(scene_name,img_idx,ImageType.RGB)
        rgb = mmcv.load(rgb_file)
        
        depth_file = self.get_image_filename(scene_name, img_idx, ImageType.MASK)
        depth = mmcv.load(depth_file)
        
        mask_file= self.get_image_filename(scene_name,img_idx, ImageType.MASK) 
        mask = mmcv.load(mask_file)
        
        pose = self.get_pose_from_scene_name_and_idx(scene_name,img_idx)
        
        return rgb, depth, mask, pose
    
    def get_random_image_index(self, scene_name):
        """return a random image index from a given scene

        Args:
            scene_name ([str]): [description]

        Returns:
            [int]: [description]
        """
        pose_data = self.get_pose_data(scene_name)
        image_idxs = pose_data.keys()
        random.choice(image_idxs)
        random_idx = random.choice(image_idxs)
        return random_idx
    
    def get_img_idx_with_different_pose(self, scene_name, pose_a, threshold = 0.2, angle_threshold = 20, num_attempts = 10):
        """
        Try to get an image with a different pose to the one passed in. If one can't be found
        then return None


        Args:
            scene_name ([type]): [description]
            pose_a ([type]): [description]
            threshold (float, optional): [description]. Defaults to 0.2.
            angle_threshold (int, optional): [description]. Defaults to 20.
            num_attempts (int, optional): [description]. Defaults to 10.

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]

        Yields:
            [type]: [description]
        """
        
        counter = 0
        while counter < num_attempts:
            img_idx = self.get_random_image_index(scene_name)
            pose = self.get_pose_from_scene_name_and_idx(scene_name, img_idx)
            
            diff = utils.compute_distance_between_poses(pose_a, pose)
            angle_diff = utils.compute_angle_between_poses(pose_a, pose)
            if(diff > threshold)  or (angle_diff > angle_threshold):
                return img_idx
            counter += 1
        
        return None
        
        





class SpartanDataSource(DenseCorrDataSource):
    
    def __init__(self, config):
        
        if config is not None:
            self._setup_scene_data(config)
        else:
            raise ValueError("You need to give a DataSource config")
        self._pose_data = dict()
        
        # self._initialize_rgb_image_to_tensor()
        
        
        self.init_length()
        print("Using SpartanDataSource")
        print("  - number of scenes", self._num_scenes)
        print("  - total images", self.num_images_total)
    
    
    def _setup_scene_data(self, config):
        """[summary]
        Initializes the data for all the different types of scenes
        
        Creates two class attributes
        
        self._single_object_scene_dict
        
        Each entry of self._single_object_scene_dict is a dict with a key 
        The values are lists of scenes
        
        self._single_object_scene_dict has (key, value) = (object_id, scene config for that object)

        self._multi_object_scene_dict has (key, value) = ("train"/"test", list of scenes)

        Note that the scenes have absolute paths here
        
        Args:
            config ([type]): [description]
        """
        
        self.dataSource_root_path = config['logs_root_path']
        
        self._single_object_scene_dict = dict()
        
        data_config_root_path = os.path.join('config','dataset')
        
        for config_file in config["single_object_scenes_config_files"]:
            config_file = os.path.join(data_config_root_path, 'single_object', config_file)
            single_object_scene_config = mmcv.load(config_file)
            object_id = single_object_scene_config["object_id"]
            
            # check if we already have this object in our datasource or not
            
            if object_id not in self._single_object_scene_dict:
                self._single_object_scene_dict[object_id] = single_object_scene_config
            else:
                existing_config = self._single_object_scene_dict[object_id]
                merged_config = SpartanDataSource.merge_single_object_configs([existing_config,single_object_scene_config])
                self._single_object_scene_dict[object_id] =merged_config
        
        # to do 
        # self._multi_object_scene_dict
        
        self._scene_structure = SceneStructure(self.dataSource_root_path)
        
        self._config = dict()
        self._config["logs_root_path"] = config['logs_root_path']
        self._config["single_object"] = self._single_object_scene_dict
        
        # self._setup_data_load_types()
        
    
    def scene_generator(self, mode=None):
        """
        Return a generator that traverses all the scenes

        Args:
            mode (string): [training mode,different mode have different generator method]. Defaults to None.
        """
        
        if mode is None:
            mode = self.mode
            
        for object_id, single_object_scene_dict in self._single_object_scene_dict.iteritems():
            for scene_name in single_object_scene_dict[mode]:
                yield scene_name
        
        for scene_name in self._multi_object_scene_dict[mode]:
            yield scene_name
    
    
    def init_length(self):
        """
        Computes the total number of images and scenes in this dataset 
        Sets the resualt to class variable self.num_images_total and self._num_scenes
        """
        
        self.num_images_total = 0
        self_num_scenes = 0
        for scene_name in self.scene_generator():
            scene_directory = self.get_full_path_for_scene(scene_name)
            rgb_images_regex = os.path.join(scene_directory, "images/*_rgb.png")
            rgb_images_regex = os.path.join(self._scene_structure.images_dir(scene_name), "*_rgb.png")
            all_rgb_images_in_scene = glob.glob(rgb_images_regex)
            num_images_this_scene = len(all_rgb_images_in_scene)
            self.num_images_total += num_images_this_scene
            self._num_scenes +=1
        
    
    def get_scene_list(self, mode=None):
        """
        Return a list of all scenes in this dataset

        Args:
            mode (srting): . Defaults to None.
        """
        
        scene_generator = self.scene_generator(mode=mode)
        scene_list = []
        for scene_name  in scene_generator:
            scene_list.append(scene_name)
        
    def get_list_of_objects(self):
        """
        Returns a list of object ids
        
        """
        return self._single_object_scene_dict.keys()
    
    def get_scene_list_for_object(self,object_id, mode=None):
        """Return list of scenes for a given object.Return  scenes

        Args:
            object_id (string): [a given object]
            mode ([string], "test" or "train"): [description]. Defaults to None.
        """
        
        if mode is None:
            mode = self.mode 
        
        return copy.copy(self._single_object_scene_dict[object_id][mode])
    

    def get_full_path_for_scene(self, scene_name):
        """return the full path to the processed logs folder

        Args:
            scene_name ([str]): [description]
        """
        return os.path.join(self.logs_root_path, scene_name,'processed')
    
    def get_random_scene_name(self):
        
        pass
    
    def get_random_single_object_scene_name(self, object_id):
        """Return a random scene name for that object 

        Args:
            object_id ([str]): [description]

        Returns:
            [str]: [description]
        """
        
        scene_list = self._single_object_scene_dict[object_id][self.mode]
        return random.choice(scene_list)
    
        
    
    def get_random_object_id_and_int(self):
        
        object_id_list = self._single_object_scene_dict.keys()
        random_object_id = random.choice(object_id_list)
        object_id_int = sorted(self._single_object_scene_dict.keys()).index(random_object_id)
        return random_object_id, object_id_int
    
    
    def get_different_scene_for_object(self, object_id, scene_name):
        """return a differrent scene name

        Args:
            object_id ([type]): [description]
            scene_name ([type]): [description]

        Raises:
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        scene_list = self._single_object_scene_dict[object_id][self.mode]
        
        if len(scene_list) == 1:
            raise ValueError("There is only one scene of this object, can't sample a different one")
        
        idx_array = np.arange(0, len(scene_list))
        rand_idxs = np.random.choice(idx_array, 2, replace=False)
        
        for idx in rand_idxs:
            scene_name_b = scene_list[idx]
            if scene_name != scene_name_b:
                return scene_name_b
        
        raise ValueError("It (should) be impossible to get here !!!!")
    

    def get_two_different_object_ids(self):
        """return two different random object ids

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        
        object_id_list = self._single_object_scene_dict.keys()
        if len(object_id_list) == 1:
            raise ValueError("There is only one object, can't sample a different one ")
        
        idx_array = np.arange(0, len(object_id_list))
        rand_idxs = np.random.choice(idx_array,2 ,replace= False)
        
        object_1_id = object_id_list[rand_idxs[0]]
        object_2_id = object_id_list[rand_idxs[1]]
        
        assert object_1_id != object_2_id
        return object_1_id, object_2_id
        
    
        
        
    def get_all_pose_data(self):
        
        """Efficiently pre-loads all pose data for the scenes. This is because when used as
        part of torch DataLoader in threaded way it behaves strangely
        """
        
        for scene_name in self.scene_generator():
            
            if scene_name not in self._pose_data:
            # log
                self._pose_data[scene_name]= self._scene_structure.get_pose_data(scene_name)
                
                
    
    def get_length(self):
        """interface for dataset to get the __len__

        Returns:
            [int]: [description]
        """
        
        
        length=0
        
        return length
    
    
    
    def get_sample(self, idx=None):
        """interface for  dataset to get the item

        Args:
            idx ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: dict() for supple the data to dataset __getitem__ methods
        """
        
        
        dataDict = dict()
        
        data_load_type = self._get_data_load_type()
        
        # Case 0: Same Scene, same object 
        if data_load_type == SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE:
            if self._verbose:
                print("Same scene, same object")
            return self.get_single_object_within_scene_data()
        
        # case 1: Same object , different scene
        if data_load_type == SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE:
            if self._verbose:
                print("Same object, different scene")
            return self.get_single_object_across_scene_data()
        
        #case 2: Different object
        if data_load_type == SpartanDatasetDataType.DIFFERENT_OBJECT:
            if self._verbose:
                print("Different object")
            return self.get_different_object_data()
        # Case 3: Multi object 
        if data_load_type == SpartanDatasetDataType.MULTI_OBJECT:
            if self._verbose:
                print("Multi object")
            return self.get_multi_object_within_scene_data()
        
        # Case 4: Synthetic multi object
        if data_load_type == SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT:
            if self._verbose:
                print("Synthetic multi object")
            return self.get_synthetic_multi_object_within_scene_data()
        
        
        
        return  dataDict
    
    
    
    #######################################################################
    #####       case 1:
    #######################################################################
    
    def get_single_object_within_scene_data(self):
        """
        """
        if self.get_number_of_unique_single_objects() == 0:
            raise ValueError("There are no single object scenes in this dataset")
        
        object_id = self.get_random_object_id()
        scene_name = self.get_random_single_object_scene_name(object_id)
        
        metadata = dict()
        metadata["object_id"] = object_id
        metadata["object_id_int"] =  sorted(self._single_object_scene_dict.keys()).index(object_id)
        metadata["type"] = SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE
        
        return self.get_within_scene_data(scene_name, metadata)
    
    
    def get_within_scene_data(self,scene_name, metadata,for_synthetic_multi_object=False):
        """The method through which the dataset is accessed for training.
        
        Each call is the result of 
        a random sampling over: 
        - random scene 
        - random rgbd frame from that scene 
        - random rgbd frame (different enough pose) from that scene 
        - various randomization in the match generation and non-match generation procedure 
        
        return a larege amount of variables, separated by commas.
        
        0th return arg: the type of data sampled (this can be used as a flag for different loss fuctions)
        0th type: string
        
        1st, 2nd return args: image_a_rgb, image_b_rgb
        1st, 2nd rtype: 3-dimensional torch.FloatTensor of shape (image_height, image_width,3)
        
        3rd, 4th return args: matches_a , matches_b
        3rd, 4th rtype: 1-dimensional torch.LongTensor of shape (num_matches)
        
        5th, 6th return args: masked_non_matches_a, masted non_matches_b
        5thm 6th rtype: 1-demensional torch.LongTensor of shape (num_non_matches)
        
        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensinal torch.LongTensor of shape (num_non_matches)
        
        9th,10th return args: blind_non_matches_a, blind_non_matches_b
        9th,10th rtype: 1-dimensional torch.LongTensor of shape(num_non_matches)
        
        11th return arg: metadata useful for plotting, and-or other flags for loss functions
        11th rtype: dict
        
        Return values 3,4,5,6,7,8,9,10 are all in the "single index" format for pixels. That is 
        
        (u,v) ----> n = u + image_width*v
        
        If no datapoints were found for some type of match or non-match then we return our "special"
        empty tensor. Note that due to the way the pytorch data loader functions cannot return an empty
        tensor like torch.FloatTensor([]). So we return SpartanDataset.empty_tensor()
        
        
        
        
        

        Args:
            scene_name ([type]): [description]
            metadata ([type]): [description]
            for_synthetic_multi_object (bool, optional): [description]. Defaults to False.
        """
        
        image_a_idx = self.get_random_image_index(scene_name)
        image_a, image_a_depth, image_a_mask, image_a_pose = self._scene_structure.get_rgbd_mask_pose(scene_name, image_a_idx)
        
        metadata['image_a_idx'] = image_a_idx
        
        # image b
        image_b_idx = self._scene_structure.get_img_idx_with_different_pose(scene_name, image_a_pose, num_attempts=50)
        metadata['image_a_idx'] = image_b_idx
        # if there not fount image_b then return the empty_data
        if image_b_idx is None:
            
            return self.return_empty_data()
        
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self._scene_structure.get_rgbd_mask_pose(scene_name, image_b_idx)
        
        
        
        
        
        
        
    
    
        
            
        
        
        
        
                
            
        
        