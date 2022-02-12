
import sys 
import numpy as np 
import random 
import warnings 
import mmcv 

import torch 


def pinhold_projection_camara_coordinates_to_image(xyz, K):
    """[summary]

    Args:
        xyz ([type]): [description]
        K ([type]): [description]

    Returns:
        [type]: [description]
    """
    full2_vec = np.matmul(K, xyz)
    u = full2_vec[0]/full2_vec[2]
    v = full2_vec[1]/full2_vec[2]
    #maybe_z2_vec = point_camera_2_frame_rdf_vec[2]
    z = full2_vec[2]
    
    return u, v, z
    


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
    fx =  K[0, 0]
    fy =  K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    u =   uv[0].astype(np.float32)
    v =   uv[1].astype(np.float32)
    X =  (u- cx)*z/fx
    Y =   (v- cy)*z/fy
    # u_v_1 =  np.ones((3,len(uv[0])))
    # np.array([uv[0], uv[1], ])
    # K_inv = np.linalg.inv(K)
    # pos = z * K_inv.dot(u_v_1)
    pos = np.vstack((X, Y, z))
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
                                     uv_a=None, num_attempts=20, device='CPU', img_a_mask=None):
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
    
    if uv_a is not None:
        num_attempts = 1
        uv1_vec = uv_a
        u1_vec = uv1_vec[0]
        v1_vec = uv1_vec[1]
    else:
        depth1_idx = np.where(img1_depth>0, 1, 0)
        if  img_a_mask is  None:
            synth_indices = depth1_idx
        else:
            mask1_idx = np.where(img_a_mask>0, 1, 0)
            synth_indices = mask1_idx&depth1_idx
        
        img1_indices = np.where(synth_indices>0)
        if img1_indices[0].size==0:
            return (None,None)
        rand_indices = np.random.randint(0,len(img1_indices[0]),[num_attempts])
        u1_vec = img1_indices[1][rand_indices]
        v1_vec = img1_indices[0][rand_indices]
        uv1_vec = (u1_vec, v1_vec)
    
    img1_depth_float = img1_depth.astype(np.float)
    DEPTH_IM_SCALE = 1000.0 #将厘米转换为米。
    depth1_vec = img1_depth_float[(v1_vec, u1_vec)]*1.0/DEPTH_IM_SCALE
    z1_vec = depth1_vec
    
    point_camera_frame_rdf_vec = pinhole_projection_image_to_camera_coordinates(uv1_vec, z1_vec, K)
    point_world_frame_rdf_vec = apply_transform(point_camera_frame_rdf_vec, img1_pose)
    point_camera_2_frame_rdf_vec = apply_transform(point_world_frame_rdf_vec,invert_transform(img2_pose))
    u2_float, v2_float, z2_float = pinhold_projection_camara_coordinates_to_image(point_camera_2_frame_rdf_vec, K)
    
    
    ###########vec2#########
    
    # Prune based on 
    # Case 2: the pixels projected into image b are outside FOV
    # u2_vec bounds should be: 0 image_width
    # v2_vec bounds should be: 0, image_height
    # do u2-based pruning
    
    u2_float_lower_bound = 0.0
    v2_float_lower_bound = 0.0
    epsilon = 1e-3
    u2_float_upper_bound = image_width*1.0-epsilon  # needs to be epsilon less!!
    v2_float_upper_bound = image_height*1.0-epsilon
    
    u2_bound_indices = np.where((u2_float>u2_float_lower_bound)&(u2_float<u2_float_upper_bound),1,0)
    v2_bound_indices = np.where((v2_float>v2_float_lower_bound)&(v2_float<v2_float_upper_bound),1,0)
    in_bound_indices = u2_bound_indices&v2_bound_indices
    syth_bound_indices = np.where(in_bound_indices>0)
    if syth_bound_indices[0].size==0:
        return (None,None)
    
    u2_float_prune  = u2_float[syth_bound_indices]
    v2_float_prune  =  v2_float[syth_bound_indices]
    z2_float_prune  =  z2_float[syth_bound_indices]
    
    u1_vec_prune =  u1_vec[syth_bound_indices]
    v1_vec_prune =  v1_vec[syth_bound_indices]
    z1_vec_prune =  z1_vec[syth_bound_indices]
    
    # float to int
    u2_vec = u2_float_prune.astype(np.int)
    v2_vec = v2_float_prune.astype(np.int)
    z2_vec = z2_float_prune
    uv2_vec = (u2_vec, v2_vec)
    
    img2_depth_float = img2_depth.astype(np.float)
    depth2_vec = img2_depth_float[(v2_vec, u2_vec)]*1.0/DEPTH_IM_SCALE
    
    occlusion_margin = 0.003
    z2_vec_nonocc = z2_vec - occlusion_margin

    depth2_nonocc_indices = np.where((depth2_vec>0)&(depth2_vec>z2_vec_nonocc),1,0)
    non_occluded_indices = np.where(depth2_nonocc_indices>0)
    if non_occluded_indices[0].size ==0:
        return (None,None)
    depth2_vec_prune = depth2_vec[non_occluded_indices]

    u2_vec = u2_vec[non_occluded_indices]
    v2_vec = v2_vec[non_occluded_indices]
    z2_vec = depth2_vec_prune
    
    u1_vec = u1_vec_prune[non_occluded_indices]
    v1_vec = v1_vec_prune[non_occluded_indices]
    z1_vec = u1_vec_prune[non_occluded_indices]
    
    uv1_vec = (u1_vec, v1_vec)
    uv2_vec = (u2_vec, v2_vec)
    
    return (uv1_vec, uv2_vec)






def numpy_rand_select_pixel(width, height, num_samples=1):
    two_rand_numbers = np.random.rand(2, num_samples)
    two_rand_numbers[0,:] = two_rand_numbers[0,:]*width
    two_rand_numbers[1,:] = two_rand_numbers[1,:]*height
    two_rand_ints    = np.floor(two_rand_numbers).astype(np.int64)
    return (two_rand_ints[0], two_rand_ints[1])

    
"""
def create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=100, img_b_mask=None):
"""
"""
    Takes in pixel matches (uv_b_matches) that correspond to matches in another image, and generates non-matches by just sampling in image space.
    Optionally, the non-matches can be sampled from a mask for image b.
    Returns non-matches as pixel positions in image b.
    Please see 'coordinate_conventions.md' documentation for an explanation of pixel coordinate conventions.
    ## Note that arg uv_b_matches are the outputs of batch_find_pixel_correspondences()

    :param uv_b_matches: tuple of torch.FloatTensors, where each FloatTensor is length n, i.e.:
        (torch.FloatTensor, torch.FloatTensor)
    :param img_b_shape: tuple of (H,W) which is the shape of the image
    (optional)
    :param num_non_matches_per_match: int
    (optional)
    :param img_b_mask: torch.FloatTensor (can be cuda or not)
        - masked image, we will select from the non-zero entries
        - shape is H x W
    :return: tuple of torch.FloatTensors, i.e. (torch.FloatTensor, torch.FloatTensor).
        - The first element of the tuple is all "u" pixel positions, and the right element of the tuple is all "v" positions
        - Each torch.FloatTensor is of shape torch.Shape([num_matches, non_matches_per_match])
        - This shape makes it so that each row of the non-matches corresponds to the row for the match in uv_a
    """
"""
        
    image_width  = img_b_shape[1]
    image_height = img_b_shape[0]
    
    if uv_b_matches == None:
        return None

    num_matches = len(uv_b_matches[0])
    
    def get_random_uv_b_non_matches():
        return numpy_rand_select_pixel(width=image_width, height= image_height,
                                         num_samples=num_matches*num_non_matches_per_match)
    
    if img_b_mask is not None:
        b_mask_indices = np.where(img_b_mask>0)
        if b_mask_indices[0].size ==0:
            print("warning, empy mask b")
            uv_b_non_matches = get_random_uv_b_non_matches()
        else:
            num_samples = num_matches*num_non_matches_per_match
            rand_numbers_b = np.random.rand(num_samples)*len(b_mask_indices[0])
            rand_indices_b = np.floor(rand_numbers_b).astype(np.int64)
            randomized_mask_b_indices = (b_mask_indices[0][rand_indices_b],b_mask_indices[1][rand_indices_b])
    else:
        uv_b_non_matches = get_random_uv_b_non_matches()
    
    # for each in uv_a, we want non-matches
    # first just randomly sample "non_matches"
    # we will later move random samples that were too close to being matches
    uv_b_non_matches =  (uv_b_non_matches[0].view(num_matches, num_non_matches_per_match), 
                         uv_b_non_matches[1].view(num_matches, num_non_matches_per_match))
    
    # uv_b_matches can now be used to make sure no "non_matches" are too close
    # to preserve tensor size, rather than pruning, we can perturb these in pixel space
    copied_uv_b_matches_0 = np.tile(uv_b_non_matches[0],(num_non_matches_per_match, 1)).T
    copied_uv_b_matches_1 = np.tile(uv_b_non_matches[1], (num_non_matches_per_match,1)).T
    
    diffs_0 =  copied_uv_b_matches_0 -uv_b_non_matches[0].astype(np.float32)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].astype(np.float32)
    
"""

# turns out to be faster to do this match generation on the CPU
# for the general size of params we expect
# also this will help by not taking up GPU memory, 
# allowing batch sizes to stay large
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


# in torch 0.3 we don't yet have torch.where(), although this
# is there in 0.4 (not yet stable release)
# for more see: https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
def where(cond, x_1, x_2):
    """
    We follow the torch.where implemented in 0.4.
    See http://pytorch.org/docs/master/torch.html?highlight=where#torch.where

    For more discussion see https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8


    Return a tensor of elements selected from either x_1 or x_2, depending on condition.
    :param cond: cond should be tensor with entries [0,1]
    :type cond:
    :param x_1: torch.Tensor
    :type x_1:
    :param x_2: torch.Tensor
    :type x_2:
    :return:
    :rtype:
    """
    cond = cond.type(dtype_float)
    return (cond * x_1) + ((1-cond) * x_2)

def pytorch_rand_select_pixel(width,height,num_samples=1):
    two_rand_numbers = torch.rand(2,num_samples)
    two_rand_numbers[0,:] = two_rand_numbers[0,:]*width
    two_rand_numbers[1,:] = two_rand_numbers[1,:]*height
    two_rand_ints    = torch.floor(two_rand_numbers).type(dtype_long)
    return (two_rand_ints[0], two_rand_ints[1])

def create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=100, img_b_mask=None):
    """
    Takes in pixel matches (uv_b_matches) that correspond to matches in another image, and generates non-matches by just sampling in image space.

    Optionally, the non-matches can be sampled from a mask for image b.

    Returns non-matches as pixel positions in image b.

    Please see 'coordinate_conventions.md' documentation for an explanation of pixel coordinate conventions.

    ## Note that arg uv_b_matches are the outputs of batch_find_pixel_correspondences()

    :param uv_b_matches: tuple of torch.FloatTensors, where each FloatTensor is length n, i.e.:
        (torch.FloatTensor, torch.FloatTensor)

    :param img_b_shape: tuple of (H,W) which is the shape of the image

    (optional)
    :param num_non_matches_per_match: int

    (optional)
    :param img_b_mask: torch.FloatTensor (can be cuda or not)
        - masked image, we will select from the non-zero entries
        - shape is H x W
     
    :return: tuple of torch.FloatTensors, i.e. (torch.FloatTensor, torch.FloatTensor).
        - The first element of the tuple is all "u" pixel positions, and the right element of the tuple is all "v" positions
        - Each torch.FloatTensor is of shape torch.Shape([num_matches, non_matches_per_match])
        - This shape makes it so that each row of the non-matches corresponds to the row for the match in uv_a
    """
    image_width  = img_b_shape[1]
    image_height = img_b_shape[0]

    if uv_b_matches == None:
        return None

    num_matches = len(uv_b_matches[0])

    def get_random_uv_b_non_matches():
        return pytorch_rand_select_pixel(width=image_width,height=image_height, 
            num_samples=num_matches*num_non_matches_per_match)

    if img_b_mask is not None:
        img_b_mask_flat = img_b_mask.view(-1,1).squeeze(1)
        mask_b_indices_flat = torch.nonzero(img_b_mask_flat)
        if len(mask_b_indices_flat) == 0:
            print("warning, empty mask b")
            uv_b_non_matches = get_random_uv_b_non_matches()
        else:
            num_samples = num_matches*num_non_matches_per_match
            rand_numbers_b = torch.rand(num_samples)*len(mask_b_indices_flat)
            rand_indices_b = torch.floor(rand_numbers_b).long()
            randomized_mask_b_indices_flat = torch.index_select(mask_b_indices_flat, 0, rand_indices_b).squeeze(1)
            uv_b_non_matches = (randomized_mask_b_indices_flat%image_width, randomized_mask_b_indices_flat/image_width)
    else:
        uv_b_non_matches = get_random_uv_b_non_matches()
    
    # for each in uv_a, we want non-matches
    # first just randomly sample "non_matches"
    # we will later move random samples that were too close to being matches
    uv_b_non_matches = (uv_b_non_matches[0].view(num_matches,num_non_matches_per_match), uv_b_non_matches[1].view(num_matches,num_non_matches_per_match))

    # uv_b_matches can now be used to make sure no "non_matches" are too close
    # to preserve tensor size, rather than pruning, we can perturb these in pixel space
    copied_uv_b_matches_0 = torch.t(uv_b_matches[0].repeat(num_non_matches_per_match, 1))
    copied_uv_b_matches_1 = torch.t(uv_b_matches[1].repeat(num_non_matches_per_match, 1))

    diffs_0 = copied_uv_b_matches_0 - uv_b_non_matches[0].type(dtype_float)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].type(dtype_float)

    diffs_0_flattened = diffs_0.contiguous().view(-1,1)
    diffs_1_flattened = diffs_1.contiguous().view(-1,1)

    diffs_0_flattened = torch.abs(diffs_0_flattened).squeeze(1)
    diffs_1_flattened = torch.abs(diffs_1_flattened).squeeze(1)


    need_to_be_perturbed = torch.zeros_like(diffs_0_flattened)
    ones = torch.zeros_like(diffs_0_flattened)
    num_pixels_too_close = 1.0
    threshold = torch.ones_like(diffs_0_flattened)*num_pixels_too_close

    # determine which pixels are too close to being matches
    need_to_be_perturbed = where(diffs_0_flattened < threshold, ones, need_to_be_perturbed)
    need_to_be_perturbed = where(diffs_1_flattened < threshold, ones, need_to_be_perturbed)

    minimal_perturb        = num_pixels_too_close/2
    minimal_perturb_vector = (torch.rand(len(need_to_be_perturbed))*2).floor()*(minimal_perturb*2)-minimal_perturb
    std_dev = 10
    random_vector = torch.randn(len(need_to_be_perturbed))*std_dev + minimal_perturb_vector
    perturb_vector = need_to_be_perturbed*random_vector

    uv_b_non_matches_0_flat = uv_b_non_matches[0].view(-1,1).type(dtype_float).squeeze(1)
    uv_b_non_matches_1_flat = uv_b_non_matches[1].view(-1,1).type(dtype_float).squeeze(1)

    uv_b_non_matches_0_flat = uv_b_non_matches_0_flat + perturb_vector
    uv_b_non_matches_1_flat = uv_b_non_matches_1_flat + perturb_vector

    # now just need to wrap around any that went out of bounds

    # handle wrapping in width
    lower_bound = 0.0
    upper_bound = image_width*1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * upper_bound

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat > upper_bound_vec, 
        uv_b_non_matches_0_flat - upper_bound_vec, 
        uv_b_non_matches_0_flat)

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat < lower_bound_vec, 
        uv_b_non_matches_0_flat + upper_bound_vec, 
        uv_b_non_matches_0_flat)

    # handle wrapping in height
    lower_bound = 0.0
    upper_bound = image_height*1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * upper_bound

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat > upper_bound_vec, 
        uv_b_non_matches_1_flat - upper_bound_vec, 
        uv_b_non_matches_1_flat)

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat < lower_bound_vec, 
        uv_b_non_matches_1_flat + upper_bound_vec, 
        uv_b_non_matches_1_flat)

    return (uv_b_non_matches_0_flat.view(num_matches, num_non_matches_per_match),
        uv_b_non_matches_1_flat.view(num_matches, num_non_matches_per_match))