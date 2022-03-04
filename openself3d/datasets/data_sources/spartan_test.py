import numpy as np 
import mmcv 
import time 
import os 

from  spartan  import SpartanDataSource, prepare_config

import correspondence_plotter


def subsample_tuple_pair(uv_a, uv_b, num_samples):
        """
        Subsamples a pair of tuples, i.e. (torch.Tensor, torch.Tensor), (torch.Tensor, torch.Tensor)
        """
        assert len(uv_a[0]) == len(uv_b[0])
        index = np.floor(np.random.rand(num_samples)*len(uv_a[0])).astype(np.int64)
        uv_a_downsampled = (uv_a[0][index], uv_a[1][index])
        uv_b_downsampled = (uv_b[0][index], uv_b[1][index])
        return uv_a_downsampled, uv_b_downsampled

scenes_dir = '/media/cyn/e45903dd-cd53-44c9-8622-c21e80814317/whl/dataset/dense-net-entire/pdc/logs_proto'
configRoot = '/home/cyn/code/openself3d/config/dense_correspondence'
dataSetconfigRoot = os.path.join(configRoot, 'dataset')
config_file = '4_shoes_all.yaml'

config = prepare_config(dataSetconfigRoot, config_file,  scenes_dir)

dataset =   SpartanDataSource(config,  mode = "train", debug = False)
train_config = mmcv.load(os.path.join(configRoot, 'training', 'training.yaml'))
dataset.set_parameters_from_training_config(train_config)

datatype, image_a_rgb_PIL, image_b_rgb_PIL, image_a_depth_numpy, \
   image_b_depth_numpy, uv_a, uv_b, masked_non_matches_a, masked_non_matches_b,  \
   background_non_matches_a, background_non_matches_b,  blind_non_matches_a, blind_non_matches_b, metadata  =  dataset.get_sample()



# downsample so can plot
num_matches_to_plot = 10
plot_uv_a, plot_uv_b = subsample_tuple_pair(uv_a, uv_b, num_samples=num_matches_to_plot)
plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long = subsample_tuple_pair(masked_non_matches_a, masked_non_matches_b, num_samples=num_matches_to_plot*3)
plot_uv_a_background_long, plot_uv_b_background_non_matches_long = subsample_tuple_pair(background_non_matches_a, background_non_matches_b, num_samples=num_matches_to_plot*3)
plot_blind_uv_a, plot_blind_uv_b = subsample_tuple_pair(blind_non_matches_a,  blind_non_matches_b, num_samples=num_matches_to_plot*10)

if uv_a is not None:
        fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                               image_b_rgb_PIL, image_b_depth_numpy,
                                                                               plot_uv_a, plot_uv_b, show=False)

        correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long,
                                                                   use_previous_plot=(fig, axes),
                                                                   circ_color='r')

        fig, axes = correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                               image_b_rgb_PIL, image_b_depth_numpy,
                                                                               plot_uv_a, plot_uv_b, show=False)

        correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_uv_a_background_long, plot_uv_b_background_non_matches_long,
                                                                   use_previous_plot=(fig, axes),
                                                                   circ_color='b')
        


correspondence_plotter.plot_correspondences_direct(image_a_rgb_PIL, image_a_depth_numpy,
                                                                   image_b_rgb_PIL, image_b_depth_numpy,
                                                                   plot_blind_uv_a, plot_blind_uv_b,
                                                                   circ_color='k', show=True)

    # # Mask-plotting city
    # import matplotlib.pyplot as plt
    # plt.imshow(np.asarray(image_a_mask))
    # plt.title("Mask of img a object pixels")
    # plt.show()

    # plt.imshow(np.asarray(image_a_mask) - 1)
    # plt.title("Mask of img a background")
    # plt.show()

    # temp = matches_a_mask.view(image_height, -1)
    # plt.imshow(temp)
    # plt.title("Mask of img a object pixels for which there was a match")
    # plt.show()

    # temp2 = (mask_a_flat - matches_a_mask).view(image_height, -1)
    # plt.imshow(temp2)
    # plt.title("Mask of img a object pixels for which there was NO match")
    # plt.show()