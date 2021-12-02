
import sys 


import numpy as np 
import random 
import warnings 

import mmcv 




def pinhole_projection_image_to_camera_coordinates(uv, z, K):
    """
    Takes a (u,v) pixel location to it's 3D location in camera frame.
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for a detailed explanation.

    :param uv: pixel location in image
    :type uv:
    :param z: depth, in camera frame
    :type z: float
    :param K: 3 x 3 camera intrinsics matrix
    :type K: numpy.ndarray
    :return: (x,y,z) in camera frame
    :rtype: numpy.array size (3,)
    """

    warnings.warn("Potentially incorrect implementation", category=DeprecationWarning)


    u_v_1 = np.array([uv[0], uv[1], 1])
    K_inv = np.linalg.inv(K)
    pos = z * K_inv.dot(u_v_1)
    return pos


def pinhold_projection_image_to_world_coordinates(uv, z, K , camera_to_world):
    
    
    
    pos_in_camera_frame = pinhole_projection_image_to_camera_coordinates(uv, z, K)
    pos_in_camera_frame_homog = np.append(pos_in_camera_frame, 1)
    pos_in_world_homog = camera_to_world.dot(pos_in_camera_frame_homog)
    return pos_in_world_homog[:3]

def pinhold_projection_world_to_image(world_pos, K, camera_to_world=None):
    
    world_pos_vec = np.append(world_pos, 1)
    
    # transform to camera frame if camera_to_world is not None
    if camera_to_world is not None:
        world_pos_vec = np.dot(np.linalg.inv(camera_to_world), world_pos_vec)
    
    scaled_pos = np.array([world_pos_vec[0]/world_pos_vec[2],world_pos_vec[1]/world_pos_vec[2],1])
    uv = np.dot(K, scaled_pos)[:2]
    return uv



# Optionally, uv_a specifies the pixels in img_a for which to find matches
# If uv_a is not set, then random correspondences are attempted to be found
def batch_find_pixel_correspondences(img_a_depth, img_a_pose, img_b_depth, img_b_pose,
                                     uv_a=None, num_attempts=20, device='CPU', img_a_mask=None, K=None):
    """Computes pixel correspondences in batch

    Args:
        img_a_depth (numpy 2d array (H x W) encoded as a unit16): depth image for image a
        img_a_pose (numpy 2d array, 4x4 (homogeneous tranform)): pose for image a, in right-down-forward optical frame
        img_b_depth ([type]): [description]
        img_b_pose ([type]): [description]
        uv_a ([each element of tuple is either an int, or a list-like], optional): [optinal arg, a tuple of (u,v) pixel positions for which to find matches]. Defaults to None.
        num_attempts (int, optional): [if random sampling, how many pixels will be _attempted_ to find matches for. 
                                       Not that this is not the same as asking for a spcific number of matches, 
                                       since many attempted matches will either be occluded or outside of field-of-view]. Defaults to 20.
        device (str, optional): [either 'CPU' or 'GPU']. Defaults to 'CPU'.
        img_a_mask ([ndarray, of shpae(H,W)], optional): [an image where each nonzero pixel will be used as a mask]. Defaults to None.
        K ([ndarray, of shape (H,W)], optional): [the camera intras]. Defaults to None.
        
    return:
        "Tuple of tuples", i.e. pixel position tuples for image a and image b (uv_a, uv_b)
        Each of these is a tuple of pixel positions
        [type]:
    """
    assert (img_a_depth.shape == img_b_depth.shape)
    image_width  = img_a_depth.shape[1]
    image_height = img_b_depth.shape[0]
    
    if img_a_mask is None:
        np.ones(img_a_mask,dtype=np.int64)
    
    
    
    
    


    