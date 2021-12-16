import correspondence_finder
import correspondence_plotter
import time 
import numpy as np
from spartan import SceneStructure


K = np.array([[1,0,0],
              [0,1,0],
              [0,0,1]
              ])
K[0,0] = 533.6422696034836 # focal x
K[1,1] = 534.7824445233571 # focal y
K[0,2] = 319.4091030774892 # principal point x
K[1,2] = 236.4374299691866 # principal point y
K[2,2] = 1.0

scenes_dir ="/Users/johnny/Code/openself3d/data"
sceneStruct= SceneStructure(scenes_dir)
scene = "2018-04-16-14-25-19"
img_a_idx = sceneStruct.get_random_image_index(scene)
img_a_rgb, img_a_depth, img_a_mask, img_a_pose = sceneStruct.get_rgbd_mask_pose(scene, 8)

img_b_idx = sceneStruct.get_img_idx_with_different_pose(scene, img_a_pose, num_attempts=50)
img_b_rgb, img_b_depth, img_b_mask, img_b_pose = sceneStruct.get_rgbd_mask_pose(scene, 163)

img_a_depth_numpy = np.asarray(img_a_depth)
img_a_mask = np.asarray(img_a_mask)
img_b_depth_numpy = np.asarray(img_b_depth)

start = time.time()
uv_a = (300, 200)
uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(img_a_depth_numpy, img_a_pose,
                                                                    img_b_depth_numpy, img_b_pose,K,num_attempts=10000,
                                                                    img1_mask=img_a_mask)
print(time.time()-start, "seconds")
if uv_a is not None:
    correspondence_plotter.plot_correspondences_direct(img_a_rgb, img_a_depth_numpy, img_b_rgb, img_b_depth_numpy, uv_a,uv_b)
else:
    print("try running this cell again, did not find a correspondence for this pixel")





