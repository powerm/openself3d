import os
import copy 
import math
from typing import Counter
from PIL import Image 
import numpy as np 
import mmcv 
import torch
import glob
import random
import logging
import utils

from torchvision import transforms

# from denseCorr import DenseCorrDataSource
#from  .correspondence_finder_old import  create_non_correspondences, batch_find_pixel_correspondences
import correspondence_augmentation
import  correspondence_finder_new
import constants

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

def flatten_uv_tensor(uv_tensor, image_width):
    """
        Flattens a uv_tensor to single dimensional tensor
        :param uv_tensor:
        :type uv_tensor:
        :return:
        :rtype:
    """
    
    return uv_tensor[1] * image_width + uv_tensor[0]


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

    def get_scene_images_number(self, scene_name):
        
        rgb_images_regex = os.path.join(self.images_dir(scene_name), "*_rgb.png")
        all_rgb_images_in_scene = glob.glob(rgb_images_regex)
        return   len(all_rgb_images_in_scene)
        
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
            img_index = utils.getPaddedString(img_idx, width = 6)
        
        return os.path.join(images_dir, img_index + file_extension)
    
    def get_camera_intrinsics(self, scene_name):
        
        return utils.CameraIntrinsics.from_yaml_file(self.camera_info_file(scene_name))
          
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
        :rtype: PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, a 4x4 numpy array
        """
        rgb_file = self.get_image_filename(scene_name,img_idx,ImageType.RGB)
        rgb = Image.open(rgb_file).convert('RGB')
        
        depth_file = self.get_image_filename(scene_name, img_idx, ImageType.DEPTH)
        depth = Image.open(depth_file)
        
        mask_file= self.get_image_filename(scene_name,img_idx, ImageType.MASK) 
        mask = Image.open(mask_file)
        
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
        #random.choice(list(image_idxs))
        random_idx = random.choice(list(image_idxs))
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

    
def merge_single_object_configs(config_list):
    """
    Given a list of single object configs, merge them. This basically concatenates
    all the fields ('train', 'test', 'logs_root_path')

    Asserts that 'object_id' is the same for all of the configs
    Asserts that `logs_root_path` is the same for all the configs
    :param config_list:
    :type config_list:
    :return: single object config
    :rtype: dict
    """
    config = config_list[0]
    logs_root_path = config['logs_root_path']
    object_id = config['object_id']

    train_scenes = []
    test_scenes = []
    evaluation_labeled_data_path = []

    for config in config_list:
        assert (config['object_id'] == object_id)
        assert (config['logs_root_path'] == logs_root_path)
        train_scenes += config['train']
        test_scenes += config['test']
        evaluation_labeled_data_path += config['evaluation_labeled_data_path']

    merged_config = dict()
    merged_config['logs_root_path'] = logs_root_path
    merged_config['object_id'] = object_id
    merged_config['train'] = train_scenes
    merged_config['test'] = test_scenes
    merged_config['evaluation_labeled_data_path'] = evaluation_labeled_data_path

    return merged_config


def  prepare_config(configRootPath, configFile, dataPath):
    """config Root Path structure
       --  dataset     
          -- composite
          -- multi_object
          -- single_object  

    Args:
        configRootPath ([type]): [description]
        configFile ([type]): [description]
        dataPath ([type]): [description]
    """
    configFileName = os.path.join(configRootPath, 'composite',configFile)
    config = mmcv.load(configFileName)
    
    merge_config = dict()
    merge_config["logs_root_path"] = dataPath
    
    single_object_scene_dict = dict()
    for config_file in config["single_object_scenes_config_files"]:
        config_file = os.path.join(configRootPath, 'single_object', config_file)
        single_object_scene_config = mmcv.load(config_file)
        object_id = single_object_scene_config["object_id"]
        # check if we already have this object in our datasource or not    
        if object_id not in  single_object_scene_dict:
            single_object_scene_dict[object_id] = single_object_scene_config
        else:
            existing_config = single_object_scene_dict[object_id]
            merged_config = merge_single_object_configs([existing_config, single_object_scene_config])
            single_object_scene_dict[object_id] =merged_config
        
    # to do 
    # self._multi_object_scene_dict
    multi_object_scene_dict ={"train": [], "test":[], "evaluation_labeled_data_path":[]}
    for config_file in config["multi_object_scenes_config_files"]:
        config_file = os.path.join(configRootPath, 'multi_object', config_file)
        multi_object_scene_config = mmcv.load(config_file)
            
        for key, val in iter(multi_object_scene_dict.items()):
            for item in multi_object_scene_config[key]:
                val.append(item)        
        
        #self._scene_structure = SceneStructure(self.dataSource_root_path)
    merge_config["single_object"] = single_object_scene_dict
    merge_config["multi_object"] =  multi_object_scene_dict
    return merge_config
    


class  SpartanDataSource(object):
    
    def __init__(self, config, mode="train", debug=False, verbose=False):
        
        self.debug = debug
        if self.debug:
            self._domain_randomize = False
            self.num_masked_non_matches_per_match = 5
            self.num_background_non_matches_per_match = 5
            self.cross_scene_num_samples = 1000
            self._use_image_b_mask_inv = True
            self.num_matching_attempts = 10000
            self.sample_matches_only_on_mask = True
        
        self._verbose = verbose
        
        if config is not None:
            self._setup_scene_data(config)
        else:
            raise ValueError("You need to give a DataSource config")
        self._pose_data = dict()
        
        if mode == "test":
            self.mode = "test"
        elif mode == "train":
            self.mode = "train"
        else:
            raise ValueError("mode should be one of [test, train")
        
        self._initialize_rgb_image_to_tensor()
        
        self.init_length()
        print("Using SpartanDataSource")
        print("   - in", self.mode, "mode")
        print("  - number of scenes", self._num_scenes)
        print("  - total images", self.num_images_total)
    
    def _initialize_rgb_image_to_tensor(self):
        """
        Sets up the RGB PIL.Image --> torch.FloatTensor transform
        :return: None
        :rtype:
        """
        norm_transform = transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
        self._rgb_image_to_tensor = transforms.Compose([transforms.ToTensor(), norm_transform])

    def get_image_mean(self):
        """
        Returns dataset image_mean
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_MEAN

        # return self.config["image_normalization"]["mean"]


        return constants.DEFAULT_IMAGE_MEAN

    def get_image_std_dev(self):
        """
        Returns dataset image std_dev
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_STD_DEV

        # return self.config["image_normalization"]["std_dev"]

        return constants.DEFAULT_IMAGE_STD_DEV
 
    def rgb_image_to_tensor(self, img):
        """
        Transforms a PIL.Image to a torch.FloatTensor.
        Performs normalization of mean and std dev
        :param img: input image
        :type img: PIL.Image
        :return:
        :rtype:
        """

        return self._rgb_image_to_tensor(img)
    
    def _setup_data_load_types(self):

        self._data_load_types = []
        self._data_load_type_probabilities = []
        if self.debug:
            #self._data_load_types.append(SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE)
            # self._data_load_types.append(SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE)
            # self._data_load_types.append(SpartanDatasetDataType.DIFFERENT_OBJECT)
            # self._data_load_types.append(SpartanDatasetDataType.MULTI_OBJECT)
            self._data_load_types.append(SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT)
            self._data_load_type_probabilities.append(1)
    
    def _setup_scene_data(self,  config):
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
        self.dataRootPath =  config['logs_root_path']
        self._scene_structure = SceneStructure(self.dataRootPath)
        self._config = config
        self._setup_data_load_types()
    
    def set_parameters_from_training_config(self, training_config):
        """
        Some parameters that are really associated only with training, for example
        those associated with random sampling during the training process,
        should be passed in from a training.yaml config file.

        :param training_config: a dict() holding params
        """

        if (self.mode == "train") and (training_config["dataset"]["domain_randomize"]):
            logging.info("enabling domain randomization")
            self._domain_randomize = True
        else:
            self._domain_randomize = False

        # self._training_config = copy.deepcopy(training_config["training"])

        self.num_matching_attempts = int(training_config['dataset']['num_matching_attempts'])
        self.sample_matches_only_on_mask = training_config['dataset']['sample_matches_only_on_mask']
        self.num_non_matches_per_match = training_config['dataset']["num_non_matches_per_match"]
        self.num_masked_non_matches_per_match     = int(training_config['dataset']["fraction_masked_non_matches"] * self.num_non_matches_per_match)
        self.num_background_non_matches_per_match = int(training_config['dataset'][
                                                    "fraction_background_non_matches"] * self.num_non_matches_per_match)
        self.cross_scene_num_samples              = training_config['dataset']["cross_scene_num_samples"]
        self._use_image_b_mask_inv = training_config["dataset"]["use_image_b_mask_inv"]

        self._data_load_types = []
        self._data_load_type_probabilities = []
        p = training_config["dataset"]["data_type_probabilities"]["SINGLE_OBJECT_WITHIN_SCENE"] 
        if p > 0:
            print("using SINGLE_OBJECT_WITHIN_SCENE")
            self._data_load_types.append(SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE)
            self._data_load_type_probabilities.append(p)

        p = training_config["dataset"]["data_type_probabilities"]["SINGLE_OBJECT_ACROSS_SCENE"]
        if p > 0:
            print("using SINGLE_OBJECT_ACROSS_SCENE")
            self._data_load_types.append(SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE)
            self._data_load_type_probabilities.append(p)

        p = training_config["dataset"]["data_type_probabilities"]["DIFFERENT_OBJECT"]
        if p > 0:
            print("using DIFFERENT_OBJECT")
            self._data_load_types.append(SpartanDatasetDataType.DIFFERENT_OBJECT)
            self._data_load_type_probabilities.append(p)

        p = training_config["dataset"]["data_type_probabilities"]["MULTI_OBJECT"]
        if p > 0:
            print("using MULTI_OBJECT")
            self._data_load_types.append(SpartanDatasetDataType.MULTI_OBJECT)
            self._data_load_type_probabilities.append(p)

        p = training_config["dataset"]["data_type_probabilities"]["SYNTHETIC_MULTI_OBJECT"]
        if p > 0:
            print("using SYNTHETIC_MULTI_OBJECT")
            self._data_load_types.append(SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT)
            self._data_load_type_probabilities.append(p)

        self._data_load_type_probabilities = np.array(self._data_load_type_probabilities)
        #self._data_load_type_probabilities /= np.sum(self._data_load_type_probabilities)
        self._data_load_type_probabilities= self._data_load_type_probabilities / np.sum(self._data_load_type_probabilities)
 
    def scene_generator(self, mode=None):
        """
        Return a generator that traverses all the scenes

        Args:
            mode (string): [training mode,different mode have different generator method]. Defaults to None.
        """
        
        if mode is None:
            mode = self.mode
            
        for object_id, single_object_scene_dict in iter(self._config['single_object'].items()):
            for scene_name in single_object_scene_dict[mode]:
                yield scene_name

        for scene_name in self._config['multi_object'][mode]:
            yield scene_name
    
    def get_scene_list(self, mode=None):
        """
        Returns a list of all scenes in this dataset
        :return:
        :rtype:
        """
        scene_generator = self.scene_generator(mode=mode)
        scene_list = []
        for scene_name in scene_generator:
            scene_list.append(scene_name)

        return scene_list
    
    
    def init_length(self):
        """
        Computes the total number of images and scenes in this dataset 
        Sets the resualt to class variable self.num_images_total and self._num_scenes
        """
        self.num_images_total = 0
        self._num_scenes = 0
        for scene_name in self.scene_generator():
            num_images_this_scene =  self._scene_structure.get_scene_images_number(scene_name)
            self.num_images_total += num_images_this_scene
            self._num_scenes +=1
    
    
    def _get_data_load_type(self):
        """
        Gets a random data load type from the allowable types
        :return: SpartanDatasetDataType
        :rtype:
        """
        return np.random.choice(self._data_load_types, 1, p=self._data_load_type_probabilities)[0]
    
        
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
    

    # def get_full_path_for_scene(self, scene_name):
    #     """return the full path to the processed logs folder

    #     Args:
    #         scene_name ([str]): [description]
    #     """
    #     return os.path.join(self.logs_root_path, scene_name,'processed')
    
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
        
        object_id_list = self._config['single_object'].keys()
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
    
    
    def return_empty_data(self, image_a_rgb, image_b_rgb, metadata=None):
        if metadata is None:
            metadata = dict()

        empty = SpartanDataSource.empty_tensor()
        return -1, image_a_rgb, image_b_rgb, empty, empty, empty, empty, empty, empty, empty, empty, metadata

    @staticmethod
    def empty_tensor():
        """
        Makes a placeholder tensor
        :return:
        :rtype:
        """
        return torch.LongTensor([-1])
    

    @staticmethod
    def mask_image_from_uv_flat_tensor(uv_flat_tensor, image_width, image_height):
        """
        Returns a torch.LongTensor with shape [image_width*image_height]. It has a 1 exactly
        at the indices specified by uv_flat_tensor
        :param uv_flat_tensor:
        :type uv_flat_tensor:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :return:
        :rtype:
        """
        image_flat = torch.zeros(image_width*image_height).long()
        image_flat[uv_flat_tensor] = 1
        return image_flat
    
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
    #####       case 1: Same scene, same object
    #######################################################################
    def get_number_of_unique_single_objects(self):
        """
        Returns the number of unique objects in this dataset with single object scenes
        :return:
        :rtype:
        """
        return len(self._config['single_object'].keys())
    
    def get_random_object_id(self):
        """
        Returns a random object_id
        :return:
        :rtype:
        """
        object_id_list = self._config['single_object'].keys()
        #return random.choice(object_id_list)
        return random.choice(list(object_id_list))
    
    def get_random_object_id_and_int(self):
        """
        Returns a random object_id (a string) and its "int" (i.e. numerical unique id)
        :return:
        :rtype:
        """
        object_id_list = self._single_object_scene_dict.keys()
        random_object_id = random.choice(object_id_list)
        object_id_int = sorted(self._single_object_scene_dict.keys()).index(random_object_id)
        return random_object_id, object_id_int
    
    def get_random_single_object_scene_name(self, object_id):
        """
        Returns a random scene name for that object
        :param object_id: str
        :type object_id:
        :return: str
        :rtype:
        """
        scene_list = self._config['single_object'][object_id][self.mode]
        return random.choice(scene_list)
    
    def get_single_object_within_scene_data(self):
        """
        """
        if self.get_number_of_unique_single_objects() == 0:
            raise ValueError("There are no single object scenes in this dataset")
        
        object_id = self.get_random_object_id()
        scene_name = self.get_random_single_object_scene_name(object_id)
        
        metadata = dict()
        metadata["object_id"] = object_id
        metadata["object_id_int"] =  sorted(self._config['single_object'].keys()).index(object_id)
        metadata["type"] = SpartanDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE
        
        return self.get_within_scene_data(scene_name, metadata)

    #######################################################################
    #####       case 2: Same object, different scene
    #######################################################################
    
    def get_different_scene_for_object(self, object_id, scene_name):
        """
        Return a different scene name
        :param object_id:
        :type object_id:
        :return:
        :rtype:
        """

        scene_list = self._config['single_object'][object_id][self.mode]
        if len(scene_list) == 1:
            raise ValueError("There is only one scene of this object, can't sample a different one")

        idx_array = np.arange(0, len(scene_list))
        rand_idxs = np.random.choice(idx_array, 2, replace=False)

        for idx in rand_idxs:
            scene_name_b = scene_list[idx]
            if scene_name != scene_name_b:
                return scene_name_b
    
    def get_single_object_across_scene_data(self):
        """
        Simple wrapper for get_across_scene_data(), for the single object case
        """
        metadata = dict()
        object_id = self.get_random_object_id()
        scene_name_a = self.get_random_single_object_scene_name(object_id)
        scene_name_b = self.get_different_scene_for_object(object_id, scene_name_a)
        metadata["object_id"] = object_id
        metadata["scene_name_a"] = scene_name_a
        metadata["scene_name_b"] = scene_name_b
        metadata["type"] = SpartanDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE
        return self.get_across_scene_data(scene_name_a, scene_name_b, metadata)    

    
    
    #######################################################################
    #####       case 3: Different object
    ####################################################################### 
    def get_different_object_data(self):
        """
        Simple wrapper for get_across_scene_data(), for the different object case
        """
        metadata = dict()
        object_id_a, object_id_b = self.get_two_different_object_ids()
        scene_name_a = self.get_random_single_object_scene_name(object_id_a)
        scene_name_b = self.get_random_single_object_scene_name(object_id_b)

        metadata["object_id_a"]  = object_id_a
        metadata["scene_name_a"] = scene_name_a
        metadata["object_id_b"]  = object_id_b
        metadata["scene_name_b"] = scene_name_b
        metadata["type"] = SpartanDatasetDataType.DIFFERENT_OBJECT
        return self.get_across_scene_data(scene_name_a, scene_name_b, metadata)  
    
    
    #######################################################################
    #####       case 4: Multi object
    ####################################################################### 
    def get_multi_object_within_scene_data(self):
        """
        Simple wrapper around get_within_scene_data(), for the multi object case
        """

        if not self.has_multi_object_scenes():
            raise ValueError("There are no multi object scenes in this dataset")

        scene_name = self.get_random_multi_object_scene_name()

        metadata = dict()
        metadata["scene_name"] = scene_name
        metadata["type"] = SpartanDatasetDataType.MULTI_OBJECT

        return self.get_within_scene_data(scene_name, metadata)  
    


    
    def get_within_scene_data(self, scene_name, metadata, for_synthetic_multi_object=False):
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
        # image a
        image_a_idx =  self._scene_structure.get_random_image_index(scene_name)
        image_a_rgb, image_a_depth, image_a_mask, image_a_pose = self._scene_structure.get_rgbd_mask_pose(scene_name, image_a_idx)
        # image b
        image_b_idx = self._scene_structure.get_img_idx_with_different_pose(scene_name, image_a_pose, num_attempts=50)
        # if there not fount image_b then return the empty_data
        if image_b_idx is None:
            logging.info("no frame with sufficiently different pose found, returning")
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self._scene_structure.get_rgbd_mask_pose(scene_name, image_b_idx)
        metadata['image_a_idx'] = image_a_idx
        metadata['image_b_idx'] = image_b_idx
        # get the camera_intrinsics
        camera_intrinsics = self._scene_structure.get_camera_intrinsics(scene_name)
        camera_intrinsics_K = camera_intrinsics.K
        # return if mask size below a threshold
        image_a_mask_numpy = np.asarray(image_a_mask)
        image_b_mask_numpy = np.asarray(image_b_mask)
        img_size = np.size(image_a_mask_numpy)
        min_mask_size = 0.01*img_size
        if(np.sum(image_a_mask_numpy)< min_mask_size) or (np.sum(image_b_mask_numpy) < min_mask_size):
            logging.info("not enough pixels in mask, skipping")
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)
        #  change depht to numpy         
        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)   
        
        if self.sample_matches_only_on_mask:
            correspondence_mask = np.asarray(image_a_mask)
        else:
            correspondence_mask = None
        
        # #########################################################################
        #################                      f ind correspondences         ##########################
        ##########################################################################
        
        uv_a, uv_b = correspondence_finder_new.batch_find_pixel_correspondences(image_a_depth_numpy, image_a_pose, 
                                                                            image_b_depth_numpy, image_b_pose,camera_intrinsics_K,
                                                                            img_a_mask = correspondence_mask,
                                                                            num_attempts=self.num_matching_attempts,
                                                                            )
        
        if for_synthetic_multi_object:
            return image_a_rgb, image_b_rgb, image_a_depth, image_b_depth, image_a_mask, image_b_mask, uv_a, uv_b, camera_intrinsics
        
        if uv_a is None: 
            logging.info("no matches found, returning")
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)
        
        # #########################################################################
        #################           data_augmentation                         ##########################
        ##########################################################################
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)
            if not self.debug:
                [image_a_rgb, image_a_depth, image_a_mask], uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_depth, image_a_mask], uv_a)
                [image_b_rgb, image_b_depth, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_b_rgb, image_b_depth, image_b_mask], uv_b)
            else:
                [image_a_rgb, image_a_depth, image_a_mask], uv_a = correspondence_augmentation.random_image_and_indices_mutation([imag_a_rgb,image_a_depth,image_a_mask], uv_a)
                [image_b_rgb, image_b_depth, image_b_mask], uv_b = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb,image_b_depth, image_b_mask], uv_b)
        
        # Update the augmentation  result
        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)
        image_a_mask_numpy = np.asarray(image_a_mask)
        image_b_mask_numpy = np.asarray(image_b_mask)
        
        # #########################################################################
        #################           find non correspondence              ##########################
        ##########################################################################
        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]
        # find mask  non-correspondence
        masked_non_matches_a, masked_non_matches_b = correspondence_finder_new.create_non_correspondences(uv_a, uv_b, image_b_shape,
                                                             num_non_matches_per_match=self.num_masked_non_matches_per_match,
                                                                            img_b_mask=image_b_mask_numpy, img_b_depth = image_b_depth_numpy)
        
        # find  background non-correspondence
        if self._use_image_b_mask_inv:
            image_b_mask_inv = 1 - image_b_mask_numpy
        else:
            image_b_mask_inv = None
        background_non_matches_a, background_non_matches_b =  correspondence_finder_new.create_non_correspondences(uv_a, uv_b, image_b_shape, num_non_matches_per_match =
                                                                                                                                                    self.num_background_non_matches_per_match, img_b_mask = image_b_mask_inv, img_b_depth =  image_b_depth_numpy)
        
        # convert PIL.Image to torch.FloatTensor
        image_a_rgb_PIL = image_a_rgb
        image_b_rgb_PIL = image_b_rgb
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)

        matches_a = flatten_uv_tensor(uv_a, image_width)
        matches_b = flatten_uv_tensor(uv_b, image_width)

        # make blind non matches
        blind_non_matches_a, blind_non_matches_b, no_blind_matches_found=  correspondence_finder_new.create_blind_non_correspondences(uv_a,image_a_mask_numpy, image_b_mask_numpy)
        if no_blind_matches_found:
            blind_non_matches_a = blind_non_matches_b = SpartanDataSource.empty_tensor()

        return metadata["type"], image_a_rgb_PIL, image_b_rgb_PIL, image_a_depth_numpy, image_b_depth_numpy, uv_a, uv_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b,  blind_non_matches_a, blind_non_matches_b, metadata

    def create_non_matches(self, uv_a, uv_b_non_matches, multiplier):
        """
        Simple wrapper for repeated code
        :param uv_a:
        :type uv_a:
        :param uv_b_non_matches:
        :type uv_b_non_matches:
        :param multiplier:
        :type multiplier:
        :return:
        :rtype:
        """
        uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                     torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1))

        uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

        return uv_a_long, uv_b_non_matches_long



    #######################################################################
    #####       case 5: Synthetic multi object
    #######################################################################
    def get_synthetic_multi_object_within_scene_data(self):
        """
        Synthetic case
        """

        object_id_a, object_id_b = self.get_two_different_object_ids()
        scene_name_a = self.get_random_single_object_scene_name(object_id_a)
        scene_name_b = self.get_random_single_object_scene_name(object_id_b)

        metadata = dict()
        metadata["object_id_a"]  = object_id_a
        metadata["scene_name_a"] = scene_name_a
        metadata["object_id_b"]  = object_id_b
        metadata["scene_name_b"] = scene_name_b
        metadata["type"] = SpartanDatasetDataType.SYNTHETIC_MULTI_OBJECT

        image_a1_rgb, image_a2_rgb, image_a1_depth, image_a2_depth, image_a1_mask, image_a2_mask, uv_a1, uv_a2 = self.get_within_scene_data(scene_name_a, metadata, for_synthetic_multi_object=True)

        if uv_a1 is None:
            logging.info("no matches found, returning")
            image_a1_rgb_tensor = self.rgb_image_to_tensor(image_a1_rgb)
            return self.return_empty_data(image_a1_rgb_tensor, image_a1_rgb_tensor)

        image_b1_rgb, image_b2_rgb, image_b1_depth, image_b2_depth, image_b1_mask, image_b2_mask, uv_b1,uv_b2 = self.get_within_scene_data(scene_name_b, metadata, for_synthetic_multi_object=True)

        if uv_b1 is None:
            logging.info("no matches found, returning")
            image_b1_rgb_tensor = self.rgb_image_to_tensor(image_b1_rgb)
            return self.return_empty_data(image_b1_rgb_tensor, image_b1_rgb_tensor)

        uv_a1 = (uv_a1[0].long(), uv_a1[1].long())
        uv_a2 = (uv_a2[0].long(), uv_a2[1].long())
        uv_b1 = (uv_b1[0].long(), uv_b1[1].long())
        uv_b2 = (uv_b2[0].long(), uv_b2[1].long())

        matches_pair_a = (uv_a1, uv_a2)
        matches_pair_b = (uv_b1, uv_b2)
        merged_rgb_1, merged_mask_1, uv_a1, uv_a2, uv_b1, uv_b2 =\
         correspondence_augmentation.merge_images_with_occlusions(image_a1_rgb, image_b1_rgb,
                                                                  image_a1_mask, image_b1_mask,
                                                                  matches_pair_a, matches_pair_b)

        if (uv_a1 is None) or (uv_a2 is None) or (uv_b1 is None) or (uv_b2 is None):
            logging.info("something got fully occluded, returning")
            image_b1_rgb_tensor = self.rgb_image_to_tensor(image_b1_rgb)
            return self.return_empty_data(image_b1_rgb_tensor, image_b1_rgb_tensor)

        matches_pair_a = (uv_a2, uv_a1)
        matches_pair_b = (uv_b2, uv_b1)
        merged_rgb_2, merged_mask_2, uv_a2, uv_a1, uv_b2, uv_b1 =\
         correspondence_augmentation.merge_images_with_occlusions(image_a2_rgb, image_b2_rgb,
                                                                  image_a2_mask, image_b2_mask,
                                                                  matches_pair_a, matches_pair_b)

        if (uv_a1 is None) or (uv_a2 is None) or (uv_b1 is None) or (uv_b2 is None):
            logging.info("something got fully occluded, returning")
            image_b1_rgb_tensor = self.rgb_image_to_tensor(image_b1_rgb)
            return self.return_empty_data(image_b1_rgb_tensor, image_b1_rgb_tensor)

        matches_1 = correspondence_augmentation.merge_matches(uv_a1, uv_b1)
        matches_2 = correspondence_augmentation.merge_matches(uv_a2, uv_b2)
        matches_2 = (matches_2[0].float(), matches_2[1].float())

        # find non_correspondences
        merged_mask_2_torch = torch.from_numpy(merged_mask_2).type(torch.FloatTensor)
        image_b_shape = merged_mask_2_torch.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]

        matches_2_masked_non_matches = \
            correspondence_finder.create_non_correspondences(matches_2,
                                                             image_b_shape,
                                                             num_non_matches_per_match=self.num_masked_non_matches_per_match,
                                                                            img_b_mask=merged_mask_2_torch)
        if self._use_image_b_mask_inv:
            merged_mask_2_torch_inv = 1 - merged_mask_2_torch
        else:
            merged_mask_2_torch_inv = None

        matches_2_background_non_matches = correspondence_finder.create_non_correspondences(matches_2,
                                                                            image_b_shape,
                                                                            num_non_matches_per_match=self.num_background_non_matches_per_match,
                                                                            img_b_mask=merged_mask_2_torch_inv)


        SD = SpartanDataset
        # convert PIL.Image to torch.FloatTensor
        merged_rgb_1_PIL = merged_rgb_1
        merged_rgb_2_PIL = merged_rgb_2
        merged_rgb_1 = self.rgb_image_to_tensor(merged_rgb_1)
        merged_rgb_2 = self.rgb_image_to_tensor(merged_rgb_2)

        matches_a = SD.flatten_uv_tensor(matches_1, image_width)
        matches_b = SD.flatten_uv_tensor(matches_2, image_width)

        # Masked non-matches
        uv_a_masked_long, uv_b_masked_non_matches_long = self.create_non_matches(matches_1, matches_2_masked_non_matches, self.num_masked_non_matches_per_match)

        masked_non_matches_a = SD.flatten_uv_tensor(uv_a_masked_long, image_width).squeeze(1)
        masked_non_matches_b = SD.flatten_uv_tensor(uv_b_masked_non_matches_long, image_width).squeeze(1)

        # Non-masked non-matches
        uv_a_background_long, uv_b_background_non_matches_long = self.create_non_matches(matches_1, matches_2_background_non_matches,
                                                                            self.num_background_non_matches_per_match)

        background_non_matches_a = SD.flatten_uv_tensor(uv_a_background_long, image_width).squeeze(1)
        background_non_matches_b = SD.flatten_uv_tensor(uv_b_background_non_matches_long, image_width).squeeze(1)


        if self.debug:
            import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
            num_matches_to_plot = 10

            print ("PRE-MERGING")
            plot_uv_a1, plot_uv_a2 = SpartanDataset.subsample_tuple_pair(uv_a1, uv_a2, num_samples=num_matches_to_plot)

            # correspondence_plotter.plot_correspondences_direct(image_a1_rgb, np.asarray(image_a1_depth),
            #                                                        image_a2_rgb, np.asarray(image_a2_depth),
            #                                                        plot_uv_a1, plot_uv_a2,
            #                                                        circ_color='g', show=True)

            plot_uv_b1, plot_uv_b2 = SpartanDataset.subsample_tuple_pair(uv_b1, uv_b2, num_samples=num_matches_to_plot)

            # correspondence_plotter.plot_correspondences_direct(image_b1_rgb, np.asarray(image_b1_depth),
            #                                                        image_b2_rgb, np.asarray(image_b2_depth),
            #                                                        plot_uv_b1, plot_uv_b2,
            #                                                        circ_color='g', show=True)

            print("MERGED")
            plot_uv_1, plot_uv_2 = SpartanDataset.subsample_tuple_pair(matches_1, matches_2, num_samples=num_matches_to_plot)
            plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long =\
                SpartanDataset.subsample_tuple_pair(uv_a_masked_long, uv_b_masked_non_matches_long, num_samples=num_matches_to_plot)

            plot_uv_a_background_long, plot_uv_b_background_non_matches_long =\
                SpartanDataset.subsample_tuple_pair(uv_a_background_long, uv_b_background_non_matches_long, num_samples=num_matches_to_plot)

            fig, axes = correspondence_plotter.plot_correspondences_direct(merged_rgb_1_PIL, np.asarray(image_b1_depth),
                                                                   merged_rgb_2_PIL, np.asarray(image_b2_depth),
                                                                   plot_uv_1, plot_uv_2,
                                                                   circ_color='g', show=False)

            correspondence_plotter.plot_correspondences_direct(merged_rgb_1_PIL, np.asarray(image_b1_depth),
                                                               merged_rgb_2_PIL, np.asarray(image_b2_depth),
                                                               plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long,
                                                               use_previous_plot=(fig, axes),
                                                               circ_color='r', show=True)

            fig, axes = correspondence_plotter.plot_correspondences_direct(merged_rgb_1_PIL, np.asarray(image_b1_depth),
                                                                   merged_rgb_2_PIL, np.asarray(image_b2_depth),
                                                                   plot_uv_1, plot_uv_2,
                                                                   circ_color='g', show=False)

            correspondence_plotter.plot_correspondences_direct(merged_rgb_1_PIL, np.asarray(image_b1_depth),
                                                               merged_rgb_2_PIL, np.asarray(image_b2_depth),
                                                               plot_uv_a_background_long, plot_uv_b_background_non_matches_long,
                                                               use_previous_plot=(fig, axes),
                                                               circ_color='b')


        return metadata["type"], merged_rgb_1, merged_rgb_2, matches_a, matches_b, masked_non_matches_a, masked_non_matches_b, background_non_matches_a, background_non_matches_b, SD.empty_tensor(), SD.empty_tensor(), metadata



    def get_across_scene_data(self, scene_name_a, scene_name_b, metadata):
        """
        Essentially just returns a bunch of samples off the masks from scene_name_a, and scene_name_b.

        Since this data is across scene, we can't generate matches.

        Return args are for returning directly from __getitem__

        See get_within_scene_data() for documentation of return args.

        :param scene_name_a, scene_name_b: Names of scenes from which to each randomly sample an image
        :type scene_name_a, scene_name_b: strings
        :param metadata: a dict() holding metadata of the image pair, both for logging and for different downstream loss functions
        :type metadata: dict()
        """

        SD = SpartanDataSource

        if self.get_number_of_unique_single_objects() == 0:
            raise ValueError("There are no single object scenes in this dataset")

        image_a_idx = self._scene_structure.get_random_image_index(scene_name_a)
        image_a_rgb,  image_a_depth,  image_a_mask, image_a_pose = self._scene_structure.get_rgbd_mask_pose(scene_name_a, image_a_idx)

        metadata['image_a_idx'] = image_a_idx

        # image b
        image_b_idx = self._scene_structure.get_random_image_index(scene_name_b)
        image_b_rgb, image_b_depth, image_b_mask, image_b_pose = self._scene_structure.get_rgbd_mask_pose(scene_name_b, image_b_idx)
        metadata['image_b_idx'] = image_b_idx

        # sample random indices from mask in image a
        num_samples = self.cross_scene_num_samples
        blind_uv_a = correspondence_finder.random_sample_from_masked_image_torch(np.asarray(image_a_mask), num_samples)
        # sample random indices from mask in image b
        blind_uv_b = correspondence_finder.random_sample_from_masked_image_torch(np.asarray(image_b_mask), num_samples)

        if (blind_uv_a[0] is None) or (blind_uv_b[0] is None):
            image_a_rgb_tensor = self.rgb_image_to_tensor(image_a_rgb)
            return self.return_empty_data(image_a_rgb_tensor, image_a_rgb_tensor)

        # data augmentation
        if self._domain_randomize:
            image_a_rgb = correspondence_augmentation.random_domain_randomize_background(image_a_rgb, image_a_mask)
            image_b_rgb = correspondence_augmentation.random_domain_randomize_background(image_b_rgb, image_b_mask)

        if not self.debug:
            [image_a_rgb, image_a_mask], blind_uv_a = correspondence_augmentation.random_image_and_indices_mutation([image_a_rgb, image_a_mask], blind_uv_a)
            [image_b_rgb, image_b_mask], blind_uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_mask], blind_uv_b)
        else:  # also mutate depth just for plotting
            [image_a_rgb, image_a_depth, image_a_mask], blind_uv_a = correspondence_augmentation.random_image_and_indices_mutation(
                [image_a_rgb, image_a_depth, image_a_mask], blind_uv_a)
            [image_b_rgb, image_b_depth, image_b_mask], blind_uv_b = correspondence_augmentation.random_image_and_indices_mutation(
                [image_b_rgb, image_b_depth, image_b_mask], blind_uv_b)

        image_a_depth_numpy = np.asarray(image_a_depth)
        image_b_depth_numpy = np.asarray(image_b_depth)

        image_b_shape = image_b_depth_numpy.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]

        blind_uv_a_flat = SD.flatten_uv_tensor(blind_uv_a, image_width)
        blind_uv_b_flat = SD.flatten_uv_tensor(blind_uv_b, image_width)

        # convert PIL.Image to torch.FloatTensor
        image_a_rgb_PIL = image_a_rgb
        image_b_rgb_PIL = image_b_rgb
        image_a_rgb = self.rgb_image_to_tensor(image_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(image_b_rgb)

        empty_tensor = SD.empty_tensor()

        if self.debug and ((blind_uv_a[0] is not None) and (blind_uv_b[0] is not None)):
            import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
            num_matches_to_plot = 10

            plot_blind_uv_a, plot_blind_uv_b = SD.subsample_tuple_pair(blind_uv_a, blind_uv_b, num_samples=num_matches_to_plot*10)

            correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_blind_uv_a, plot_blind_uv_b,
                                                                   circ_color='k', show=True)

        return metadata["type"], image_a_rgb, image_b_rgb, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, empty_tensor, blind_uv_a_flat, blind_uv_b_flat, metadata
    
    
    
    
    
    
    
    
    
    