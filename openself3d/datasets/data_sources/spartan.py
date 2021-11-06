import os
import copy 
import math 
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
        
        
        
                
            
        
        