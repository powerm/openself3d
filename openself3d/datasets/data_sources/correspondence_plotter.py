import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_correspondences(images, uv_a, uv_b, use_previous_plot=None, circ_color='g',show=True):
    
    if use_previous_plot is None:
        fig, axes = plt.subplots(nrows=2, ncols=3)
    else:
        fig, axes = use_previous_plot[0], use_previous_plot[1]
    
    fig.set_figheight(10)
    fig.set_figwidth(15)
    pixel_locs = [uv_a, uv_a, uv_a, uv_b, uv_b, uv_b]
    axes = axes.flat[0:]
    # if use_previous_plot is not None:
    #     axes = [axes[1], axes[3]]
    #     images = [images[1], images[3]]
    #     pixel_locs = [pixel_locs[1], pixel_locs[3]]

    idx = 0
    for ax, img, pixel_locs in zip(axes[0:], images, pixel_locs):
        ax.set_aspect('equal')
        ax.set_title(idx)
        if img is not None:
            if isinstance(pixel_locs[0], int) or isinstance(pixel_locs[0], float):
                circ = Circle(pixel_locs, radius=3, facecolor=circ_color, edgecolor='white',fill=True, linewidth=0.5, linestyle='solid')
                ax.add_patch(circ)
            else:
                for x,y in zip(pixel_locs[0],pixel_locs[1]):
                    circ = Circle((x,y), radius=3, facecolor=circ_color, edgecolor='white',fill=True, linewidth=0.5, linestyle='solid')
                    ax.add_patch(circ)
            
            ax.imshow(img)
            idx+=1
    if show:
        plt.show()
        return None
    else:
        return fig, axes

def plot_correspondences_from_dir(log_dir, img_a, img_b, uv_a,uv_b, use_previous_plot=None, circ_color='g', show=True):
    img1_filename = log_dir+"/images"+img_a+"_rgb.png"
    img2_filename = log_dir+"/images"+img_b+"_rgb.png"
    img1_depth_filename = log_dir+"/images"+img_a+"_depth.png"
    img2_depth_filename = log_dir+"/images"+img_a+"_depth.png"
    images=[img1_filename, img2_filename, img1_depth_filename, img2_depth_filename]
    images = [mpimg.imread(x) for x in images]
    return plot_correspondences(images, uv_a, uv_b, use_previous_plot=use_previous_plot,circ_color=circ_color,show=show)

def plot_correspondences_direct(img_a_rgb, img_a_depth, img_b_rgb, img_b_depth,uv_a, uv_b,img_a_mask=None,img_b_mask=None,use_previous_plot=None, circ_color='g', show=True):
    
    
    images = [img_a_rgb, img_a_depth, img_a_mask, img_b_rgb, img_b_depth, img_b_mask]
    return plot_correspondences(images, uv_a, uv_b, use_previous_plot=use_previous_plot, circ_color=circ_color, show=show)
        

    