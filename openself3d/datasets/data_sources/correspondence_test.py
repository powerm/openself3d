import numpy as np 
import torch 
from numpy.linalg import inv 
import random 
import mmcv
from PIL  import Image
import os
import sys
import  time

import correspondence_finder_new
import correspondence_plotter
from spartan import SceneStructure

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

scenes_dir = '/media/cyn/e45903dd-cd53-44c9-8622-c21e80814317/whl/dataset/dense-net-entire/pdc/logs_proto'
sceneStruct= SceneStructure(scenes_dir)
scene="2018-04-16-14-25-19"
img_a_idx = sceneStruct.get_random_image_index(scene)
img_a_rgb, img_a_depth, img_a_mask, img_a_pose = sceneStruct.get_rgbd_mask_pose(scene, 8)
img_b_idx = sceneStruct.get_img_idx_with_different_pose(scene, img_a_pose, num_attempts=50)
img_b_rgb, img_b_depth, img_b_mask, img_b_pose = sceneStruct.get_rgbd_mask_pose(scene,  img_b_idx)

camera_intrinsics= sceneStruct.get_camera_intrinsics(scene)
K = camera_intrinsics.K

img_a_mask_numpy = np.asarray(img_a_mask)
img_a_depth_numpy  = np.asarray(img_a_depth)
img_b_mask_numpy = np.asarray(img_b_mask)
img_b_depth_numpy  = np.asarray(img_b_depth)

start = time.time()
uv_a = (300, 200)
uv_a, uv_b = correspondence_finder_new.batch_find_pixel_correspondences(img_a_depth_numpy, img_a_pose,
                                                                    img_b_depth_numpy, img_b_pose,K, img_a_mask=img_a_mask_numpy, num_attempts=5000
                                                                
                                                                    )
# print(time.time()-start, "seconds")
# if uv_a is not None:
#     correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, uv_a,uv_b)
# else:
#     print("try running this cell again, did not find a correspondence for this pixel")



# u_b_matches = torch.Tensor([50, 78, 100])
# v_a_matches = torch.Tensor([ 60, 200, 168])

# uv_b_matches =(u_b_matches,  v_a_matches)

# find non_correspondences


img_b_shape = img_b_depth_numpy.shape
image_width = img_b_shape[1]
image_height = img_b_shape[0]



num_matches = len(uv_b[0])

start2 = time.time()
uv_b_non_matches =correspondence_finder_new.create_non_correspondences(uv_b, img_b_shape, num_non_matches_per_match=4, img_b_mask=img_b_mask_numpy, img_b_depth=img_b_depth_numpy)
print("found_non_corre",time.time()-start2, "seconds")



fig, axes = correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, uv_a, uv_b,img_a_mask=img_a_mask_numpy, img_b_mask=img_b_mask_numpy, show=False)

uv_a_long = ((np.tile(uv_a[0], (4,1)).T).reshape(-1), (np.tile(uv_a[1], (4,1)).T).reshape(-1))
uv_b_non_matches_long = (uv_b_non_matches[0].reshape(-1), uv_b_non_matches[1].reshape(-1) )
fig2, axes2 = correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, uv_a_long, uv_b_non_matches_long,img_a_mask=img_a_mask_numpy,img_b_mask=img_b_mask_numpy, use_previous_plot=(fig,axes),
                                                  circ_color='r', show = False)


# img_b_mask_inv = 1 - img_b_mask_numpy
# ub_b_non_background  = correspondence_finder_new.create_non_correspondences(uv_b, img_b_shape, num_non_matches_per_match= 30, img_b_mask=img_b_mask_inv, img_b_depth=img_b_depth_numpy)
# uv_a_long_2 = ((np.tile(uv_a[0], (30,1)).T).reshape(-1), (np.tile(uv_a[1], (30,1)).T).reshape(-1))
# uv_b_non_matches_long_2 = (ub_b_non_background[0].reshape(-1), ub_b_non_background[1].reshape(-1) )
# correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, uv_a_long_2, uv_b_non_matches_long_2,img_a_mask=img_a_mask_numpy,img_b_mask=img_b_mask_numpy, use_previous_plot=(fig2,axes2),
#                                                   circ_color='b')


blind_non_matches_a, blind_non_matches_b,num = correspondence_finder_new.create_blind_non_correspondences(uv_a,img_a_mask_numpy, img_b_mask_numpy)

correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, blind_non_matches_a, blind_non_matches_b,img_a_mask=img_a_mask_numpy,img_b_mask=img_b_mask_numpy, use_previous_plot=(fig2,axes2),
                                                    circ_color='b')



