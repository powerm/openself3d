
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

def random_sample_from_masked_image_numpy(img_mask, num_samples):
    
    image_height, image_width = img_mask.shape
    
    if isinstance(img_mask, np.ndarray):
        mask_indices = np.where(img_mask>0)
        rand_indices= np.random.randint(0, len(mask_indices[0]),[num_samples])
        u_vec = mask_indices[0][rand_indices]
        v_vec = mask_indices[1][rand_indices]
        #uv_vec = np.vstack((u_vec, v_vec))
        return (u_vec, v_vec)


def apply_transform(vec3, transform4):
    ones_row = np.ones_like(vec3[0,:]).reshape(1,-1)
    ones_row= ones_row.astype(np.float)
    vec4 = np.vstack((vec3, ones_row))
    vec4 = np.matmul(transform4, vec4)
    return vec4[0:3]
    
def invert_transform(transform4):
    transform4_copy = np.copy(transform4)
    R = transform4_copy[0:3,0:3]
    R = np.transpose(R)
    transform4_copy[0:3,0:3] = R
    t = transform4_copy[0:3,3]
    inv_t = -1.0*np.transpose(R).dot(t)
    transform4_copy[0:3, 3] = inv_t
    return transform4_copy

        
def get_body_to_rdf():
    body_to_rdf = np.zeros((3,3))
    body_to_rdf[0, 1] = -1.0
    body_to_rdf[1, 2] = -1.0
    body_to_rdf[2, 0] = 1.0
    return body_to_rdf
            



# Optionally, uv_a specifies the pixels in img_a for which to find matches
# If uv_a is not set, then random correspondences are attempted to be found
def batch_find_pixel_correspondences(img1_depth, img1_pose, img2_depth, img2_pose, K,
                                     uv_a=None, num_attempts=20, device='CPU', img1_mask=None):
    """Computes pixel correspondences in batch

    Args:
        img1_depth (numpy 2d array (H x W) encoded as a unit16): depth image for image a
        img1_pose (numpy 2d array, 4x4 (homogeneous tranform)): pose for image a, in right-down-forward optical frame
        img2_depth ([type]): [description]
        img2_pose ([type]): [description]
        uv_a ([each element of tuple is either an int, or a list-like], optional): [optinal arg, a tuple of (u,v) pixel positions for which to find matches]. Defaults to None.
        num_attempts (int, optional): [if random sampling, how many pixels will be _attempted_ to find matches for. 
                                       Not that this is not the same as asking for a spcific number of matches, 
                                       since many attempted matches will either be occluded or outside of field-of-view]. Defaults to 20.
        device (str, optional): [either 'CPU' or 'GPU']. Defaults to 'CPU'.
        img1_mask ([ndarray, of shpae(H,W)], optional): [an image where each nonzero pixel will be used as a mask]. Defaults to None.
        K ([ndarray, of shape (H,W)], optional): [the camera intras]. Defaults to None.
        
    return:
        "Tuple of tuples", i.e. pixel position tuples for image a and image b (uv_a, uv_b)
        Each of these is a tuple of pixel positions
        [type]:
    """
    assert (img1_depth.shape == img2_depth.shape)
    image_width  = img1_depth.shape[1]
    image_height = img1_depth.shape[0]
    
    mask1_idx = np.where(img1_mask>0,1,0)
    depth1_idx = np.where(img1_depth>0,1,0)
    synth_indices = mask1_idx&depth1_idx
    img1_indices = np.where(synth_indices>0)
    if img1_indices[0].size==0:
        return (None,None)
    
    rand_indices = np.random.randint(0,len(img1_indices[0]),[num_attempts])
    u1_vec = img1_indices[0][rand_indices]
    v1_vec = img1_indices[1][rand_indices]
    uv1_vec = (u1_vec, v1_vec)
    img1_depth_float = img1_depth.astype(np.float)
    DEPTH_IM_SCALE = 1000.0 #将厘米转换为米。
    depth1_vec = img1_depth_float[uv1_vec]*1.0/DEPTH_IM_SCALE
    # u*Z, v*Z 
    u1_vec_Z = u1_vec.astype(np.float)*depth1_vec
    v1_vec_Z = v1_vec.astype(np.float)*depth1_vec
    z1_vec = depth1_vec
    full_vec_float = np.vstack((u1_vec_Z,v1_vec_Z, z1_vec))
    
    # tranlation
    K_inv = np.linalg.inv(K)
    point_camera_frame_rdf_vec = np.matmul(K_inv,full_vec_float)
    point_world_frame_rdf_vec = apply_transform(point_camera_frame_rdf_vec, img1_pose)
    point_camera_2_frame_rdf_vec = apply_transform(point_world_frame_rdf_vec,invert_transform(img2_pose))
    
    ###########vec2#########
    full2_vec = np.matmul(K, point_camera_2_frame_rdf_vec)
    u2_vec_float = full2_vec[0]/full2_vec[2]
    v2_vec_float = full2_vec[1]/full2_vec[2]
    maybe_z2_vec = point_camera_2_frame_rdf_vec[2]
    z2_vec = full2_vec[2]
    
    # Prune based on 
    # Case 2: the pixels projected into image b are outside FOV
    # u2_vec bounds should be: 0 image_width
    # v2_vec bounds should be: 0, image_height
    # do u2-based pruning
    
    u2_vec_lower_bound = 0.0
    v2_vec_lower_bound = 0.0
    epsilon = 1e-3
    u2_vec_upper_bound = image_height*1.0-epsilon  # needs to be epsilon less!!
    v2_vec_upper_bound = image_width*1.0-epsilon
    
    u2_bound_indices = np.where((u2_vec_float>u2_vec_lower_bound)&(u2_vec_float<u2_vec_upper_bound),1,0)
    v2_bound_indices = np.where((v2_vec_float>v2_vec_lower_bound)&(v2_vec_float<v2_vec_upper_bound),1,0)
    in_bound_indices = u2_bound_indices&v2_bound_indices
    syth_indices = np.where(in_bound_indices>0)
    if syth_indices[0].size==0:
        return (None,None)
    
    u2_vec_float_prune = u2_vec_float[syth_indices]
    v2_vec_float_prune = v2_vec_float[syth_indices]
    z2_vec_prune = z2_vec[syth_indices]
    u1_vec = u1_vec[syth_indices]
    v1_vec = v1_vec[syth_indices]
    z1_vec = z1_vec[syth_indices]
    # float to int
    u2_vec = u2_vec_float_prune.astype(np.int)
    v2_vec = v2_vec_float_prune.astype(np.int)
    uv2_vec = (u2_vec, v2_vec)
    
    img2_depth_float = img2_depth.astype(np.float)
    depth2_vec = img2_depth_float[uv2_vec]*1.0/DEPTH_IM_SCALE
    
    occlusion_margin = 0.003
    z2_vec_prune = z2_vec_prune - occlusion_margin

    depth2_01 = np.where((depth2_vec>0)&(depth2_vec>z2_vec_prune),1,0)
    non_occluded_indices = np.where(depth2_01>0)
    if non_occluded_indices.size ==0:
        return (None,None)
    depth2_vec_prune = depth2_vec[non_occluded_indices]

    u2_vec = u2_vec[non_occluded_indices]
    v2_vec = v2_vec[non_occluded_indices]
    z2_vec = depth2_vec_prune[non_occluded_indices]
    
    u1_vec = u1_vec[non_occluded_indices]
    u1_vec = v1_vec[non_occluded_indices]
    z1_vec = z1_vec[non_occluded_indices]
    
    uv1_vec = (u1_vec, v1_vec)
    uv2_vec = (u2_vec, u2_vec)
    return (uv1_vec, uv2_vec)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    