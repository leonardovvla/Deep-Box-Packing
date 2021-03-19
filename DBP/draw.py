"""

Deep Box Packing - Draw: File from Logging and Visualizing results from DBP

Author: Leonardo Albuquerque - ETH Zürich, 2021

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import numpy as np
import matplotlib
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

from pylab import *
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import tkinter
import random

# ==================================================================================================== #
# ===================================== VISUALIZATION FUNCTIONS ====================================== #
# ==================================================================================================== #

# This class is for drawing the arrows that label the orientations of the Probability map
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# This function is called by watch.py and is used for displaying the step-by-step visualization of a particular container packing & probability map
def visual_demo(packed_boxes, packed_rows, packed_cols, packed_base, packed_box_matrix, packed_probs, packed_ptr, c_w, c_l, c_h, x, y, z, box_colors, scale):

    # ======================================================== PREPARATION ======================================================== #

    matplotlib.use('TkAgg')

    # Number of boxes in the Information window
    vis_seq_size = int(packed_box_matrix.shape[2]/6)

    # Normalize probability maps for better visualization with imshow:
    # we want the brightest to be the highest probability of a given probability vector, not necessarily 1
    winner_sample = np.max(packed_probs, axis=1)
    norm_probs = packed_probs/winner_sample.reshape(packed_probs.shape[0],1)
    probs = norm_probs.reshape(norm_probs.shape[0], c_w, c_l, packed_box_matrix.shape[2])

    # Take every box in the Information window in its original orientation
    box_matrix = packed_box_matrix[:,:,:vis_seq_size]

    # Open a figure and divide it in two halves (one for the probability map and one for the container)
    fig = matplotlib.pyplot.figure(figsize=(15,5))
    canvas = fig.add_gridspec(1, 2, left=0.055, bottom=0.105, top=0.938, right=0.983, wspace=0.033, hspace=0.2)

    # Probability map
    box_maps = canvas[0].subgridspec(7, vis_seq_size+1, hspace=0.1, wspace=0.1)
    # Container
    container = canvas[1].subgridspec(1, 1)

    # This vector holds the 3D representation of the boxes in the Information window
    # (the labels of top of each column of the probability map)
    box_plh = []
    # This vector holds the mini-heatmaps at each cell of the probability map
    map_plh = []

    # For every grid cell in the Probability map:
    for orientation in range(7):
        for box in range(box_matrix.shape[2]+1):
            
            # In the first row, we need 3D plots (this is where boxes from the Information window are going to be)
            if(orientation==0 and box!=0):
                bm = fig.add_subplot(box_maps[orientation, box], projection='3d')
                bm.set(xticks=[], yticks=[])
                bm.grid(False)
                bm.set_axis_off()
                box_plh.append(bm)
                bm.set_xlim(0,4)
                bm.set_ylim(4,0)
                bm.set_zlim(0,4)

            # In the first column, we also need 3D plots (this is where the arrows for the orientation label is going to be)
            elif(box==0 and orientation!=0):

                bm = fig.add_subplot(box_maps[orientation, box], projection='3d')
                bm.set(xticks=[], yticks=[])
                bm.grid(False)
                bm.set_xlim(0,4)
                bm.set_ylim(4,0)
                bm.set_zlim(0,4)

                # For each different orientation, show its correct corresponding sequence of rotations
                # (these are static across each step, so we can already plot them)
                if(orientation==2):

                    arrow = Arrow3D([0, 4], [0, 0], [4, 0], mutation_scale=9, lw=1, arrowstyle="-|>", color="#0b55b5", connectionstyle="arc3,rad=-0.2")
                    bm.add_artist(arrow)

                elif(orientation==3):

                    arrow = Arrow3D([0, 4], [4, 0], [0, 0], mutation_scale=9, lw=1, arrowstyle="-|>", color="#0b55b5", connectionstyle="arc3,rad=0.2")
                    bm.add_artist(arrow)

                elif(orientation==4):

                    arrow1 = Arrow3D([0, 4], [4, 0], [0, 0], mutation_scale=9, lw=1, arrowstyle="-|>", color="#0b55b5", connectionstyle="arc3,rad=0.2")
                    arrow2 = Arrow3D([0, 0], [0, 4], [4, 0], mutation_scale=9, lw=1, arrowstyle="-|>", color="#0b55b5", connectionstyle="arc3,rad=0.2")
                    bm.add_artist(arrow1)
                    bm.add_artist(arrow2)

                elif(orientation==5):

                    arrow1 = Arrow3D([4, 0], [0, 4], [0, 0], mutation_scale=9, lw=1, arrowstyle="-|>", color="#0b55b5", connectionstyle="arc3,rad=-0.2")
                    arrow2 = Arrow3D([0, 0], [4, 0], [0, 4], mutation_scale=9, lw=1, arrowstyle="-|>", color="#0b55b5", connectionstyle="arc3,rad=-0.2")
                    bm.add_artist(arrow1)
                    bm.add_artist(arrow2)

                elif(orientation==6):

                    arrow = Arrow3D([0, 0], [0, 4], [4, 0], mutation_scale=9, lw=1, arrowstyle="-|>", color="#0b55b5", connectionstyle="arc3,rad=0.2")
                    bm.add_artist(arrow)

            # In every other position of the Probability map, just setup a subplot for the heatmaps
            else:
                if(box!=0 and orientation!=0):
                    bm = fig.add_subplot(box_maps[orientation, box])
                    bm.set(xticks=[], yticks=[])
                    map_plh.append(bm)

    # Prepare the container plot
    ax = fig.add_subplot(container[0,0], projection='3d')

    # Prepare scale projection
    def short_proj():
      return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj=short_proj

    # Check the number of packed boxes
    end_of_container_indices = np.where(packed_boxes[0]==0)
    if(not end_of_container_indices[0].size):
        end_of_container = packed_boxes.shape[0]
    else:
        end_of_container = np.min(end_of_container_indices)

    # Set container boundaries
    ax.set_xlim(0,c_l)
    ax.set_ylim(c_w,0)
    ax.set_zlim(0,c_h)

    # Prepare voxel grid
    voxels = (x<0) & (y<0) & (z<0)
    # Prepare color grid
    colors = np.empty(voxels.shape, dtype=object)

    # Configure a continuously updating plot
    matplotlib.pyplot.ion()

    # ======================================================== VISUALIZATION ======================================================== #

    # The steps are controlled by the user by pressing the "Enter" button.
    print("Init - Press Enter to see the next box being placed in the container according to the distribution maps")

    # Check the first sampled box
    sampled_box = (packed_ptr[0]%packed_box_matrix.shape[2])%vis_seq_size
    # Check the first sampled orientation
    sampled_orientation = int((packed_ptr[0]%packed_box_matrix.shape[2])/vis_seq_size)

    # Plot the first Probability map:

    # For every cell in the grid besides the orientation labels:
    for orientation in range(7):
        for box in range(1, box_matrix.shape[2]+1):

            # In the first row, plot every box in the Information window (in their original orientations)
            if(orientation==0):

                # Construct the box by its boundary conditions
                cube_box = (x < box_matrix[0, 2, box-1]) & (x >= 0) & (y < box_matrix[0, 1, box-1]) & (y >= 0) & (z < box_matrix[0, 3, box-1]) & (z >= 0)
                
                # If this is the sampled box, paint it red, otherwise paint it blue
                if((box-1)==sampled_box):
                    color = '#ee0101'
                else:
                    color = '#0b55b5'

                # Clear the voxel grid and add the box
                box_plh[box-1].clear()
                box_plh[box-1].set_axis_off()
                box_plh[box-1].set_xlim(0,4)
                box_plh[box-1].set_ylim(4,0)
                box_plh[box-1].set_zlim(0,4)
                box_plh[box-1].voxels(cube_box, facecolors=color, alpha=1.0, shade=False, linewidth=0.01)

            # In every other cell:    
            else:

                # Show the appropriate section of the probability map corresponding to this specific box-orientation combination
                index = (orientation-1)*box_matrix.shape[2]+(box-1)
                map_plh[index].clear()
                map_plh[index].set(xticks=[], yticks=[])
                map_plh[index].imshow(probs[0,:,:,index], vmin=0, vmax=1)

                # If this is the sampled box-orientation combination, encapsulate the heatmap with a red border
                if((orientation-1)==sampled_orientation and (box-1)==sampled_box):
                    autoAxis = map_plh[index].axis()
                    rec = Rectangle((autoAxis[0]-0.6,autoAxis[2]+0.5),(autoAxis[1]-autoAxis[0])+1.4,(autoAxis[3]-autoAxis[2])-0.9,fill=False,lw=2, color='#ee0101')
                    rec = map_plh[index].add_patch(rec)
                    rec.set_clip_on(False)

    # Update the plot
    matplotlib.pyplot.show()

    # Wait for user to press "Enter"
    next = input()

    # Until all packed boxes have been added to the container:
    for box_index in range(end_of_container):

        # Specify which box is being packed (chronological order)
        print("Box: ", box_index)

        # Add the next sampled box to the voxel grid
        if(np.sum(packed_boxes[box_index])):
            cube = (x < packed_boxes[box_index, 1]+packed_cols[box_index]) & (x >= packed_cols[box_index]) & (y < packed_boxes[box_index, 0]+packed_rows[box_index]) & (y >= packed_rows[box_index]) & (z < packed_boxes[box_index, 2]+packed_base[box_index]) & (z >= packed_base[box_index])
            voxels = voxels | cube
            colors[cube] = box_colors[box_index%49]

        # Check the next sampled box and orientation
        sampled_box = (packed_ptr[box_index+1]%packed_box_matrix.shape[2])%vis_seq_size
        sampled_orientation = int((packed_ptr[box_index+1]%packed_box_matrix.shape[2])/vis_seq_size)

        # Until the previous to last box (when the last box is packed, there is no more probability map to show):
        if(box_index < end_of_container-1):

            # For every cell in the grid besides the orientation labels:
            for orientation in range(7):
                for box in range(1, box_matrix.shape[2]+1):

                    # In the first row, plot every box in the Information window (in their original orientations)
                    if(orientation==0):

                        # Construct the box by its boundary conditions
                        cube_box = (x < box_matrix[box_index+1, 2, box-1]) & (x >= 0) & (y < box_matrix[box_index+1, 1, box-1]) & (y >= 0) & (z < box_matrix[box_index+1, 3, box-1]) & (z >= 0)
                        
                        # If this is the next sampled box, paint it red, otherwise paint it blue
                        if((box-1)==sampled_box):
                            color = '#ee0101'
                        else:
                            color = '#0b55b5'

                        # Clear the last box in the voxel grid and add the new one    
                        box_plh[box-1].clear()
                        box_plh[box-1].set_axis_off()
                        box_plh[box-1].set_xlim(0,4)
                        box_plh[box-1].set_ylim(4,0)
                        box_plh[box-1].set_zlim(0,4)
                        box_plh[box-1].voxels(cube_box, facecolors=color, alpha=1.0, shade=False, linewidth=0.01)

                    # In every other cell: 
                    else:

                        # Show the appropriate section of the probability map corresponding to this specific box-orientation combination
                        index = (orientation-1)*box_matrix.shape[2]+(box-1)
                        map_plh[index].clear()
                        map_plh[index].set(xticks=[], yticks=[])
                        map_plh[index].imshow(probs[box_index+1,:,:,index], vmin=0, vmax=1)

                        # If this is the next sampled box-orientation combination, encapsulate the heatmap with a red border
                        if((orientation-1)==sampled_orientation and (box-1)==sampled_box):
                            autoAxis = map_plh[index].axis()
                            rec = Rectangle((autoAxis[0]-0.6,autoAxis[2]+0.5),(autoAxis[1]-autoAxis[0])+1.4,(autoAxis[3]-autoAxis[2])-0.9,fill=False,lw=2, color='#ee0101')
                            rec = map_plh[index].add_patch(rec)
                            rec.set_clip_on(False)

        # Plot container voxels
        ax.voxels(voxels, facecolors=colors, alpha=0.9, shade=False, linewidth=0.01)
    
        # Update the plot
        matplotlib.pyplot.show()

        # Wait for the user to press "Enter"
        next = input()

    print("End of visual demonstration!")

    # Wait for a last "Enter" press before ending the visualization
    next = input()
    return

# This functions is called in the Validation procedure of DBP and saves packing info + images of the packed containers for posterior analysis
def draw_container(draw_info, V_rew, path, **kwargs):

    # Information for text file
    packed_factors = draw_info[7]

    # Information for container rendering and posterior visual demo analysis
    packed_boxes = np.asarray(draw_info[0])
    packed_rows = np.asarray(draw_info[1])
    packed_cols = np.asarray(draw_info[2])
    packed_base = np.asarray(draw_info[3])
    packed_box_matrix = np.asarray(draw_info[4])
    packed_probs = np.asarray(draw_info[5])
    packed_ptr = np.asarray(draw_info[6])

    # Scale Projection under three axes
    def short_proj():
      return np.dot(Axes3D.get_proj(ax), scale)

    y_scale=1.0
    x_scale=kwargs['container_length']/kwargs['container_width']
    z_scale=kwargs['container_height']/kwargs['container_width']

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    # Create axes measures
    x, y, z = np.indices((kwargs['container_length'], kwargs['container_width'], kwargs['container_height']))

    # Palette of colors for coloring boxes
    box_colors = ['#0b55b5', '#ffa600', '#4bab3c', '#e0e048', '#cc2f2f', '#50bdc9', 
                  '#a47c5f', '#321f91', '#edecf9', '#dd9286',  '#91aec6', 'silver',
                  'khaki', '#ada5de', 'coral', '#ffd64d', '#6e6702', 'yellowgreen',
                  'lightblue', 'salmon', '#004fa4', 'tan', '#fdff04', 'lavender', '#037272', 
                  '#eb1629', '#2a3132', '#336b87', '#f27657', '#598234', '#e53a3a', '#aebd38', '#c4dfe6',
                  '#44b8b1', '#ffbb00', '#fb6542', '#375e97', '#4cb5f5', '#f4cc70', '#1e434c', '#9b4f0f',
                  '#f1f1f2', '#bcbabe', '#20c0df', '#011a27', '#eed8ae', '#f0810f', '#4b7447', '#2d4262']

    # For every container:
    for container_index in range(packed_boxes.shape[1]):

        # =================== DRAWING =================== #

        # Open a 3D plot
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Setup axes limits
        ax.set_xlim(0,kwargs['container_length'])
        ax.set_ylim(kwargs['container_width'],0)
        ax.set_zlim(0,kwargs['container_height'])

        # Apply scale
        ax.get_proj=short_proj

        # Initialize voxel grid
        voxels = (x<0) & (y<0) & (z<0)

        # Initialize color grid 
        colors = np.empty(voxels.shape, dtype=object)

        # For every box in this container:
        for box_index in range(packed_boxes.shape[0]):

            # If there is a box listed in packed_boxes:
            if(np.sum(packed_boxes[box_index, container_index])):
                # Construct the box by stating its boundary conditions
                cube = (x < packed_boxes[box_index, container_index, 1]+packed_cols[box_index, container_index]) & (x >= packed_cols[box_index, container_index]) & (y < packed_boxes[box_index, container_index, 0]+packed_rows[box_index, container_index]) & (y >= packed_rows[box_index, container_index]) & (z < packed_boxes[box_index, container_index, 2]+packed_base[box_index, container_index]) & (z >= packed_base[box_index, container_index])
                # Mask it in the voxel grid
                voxels = voxels | cube
                # Select its color and insert it in the color grid
                colors[cube] = box_colors[box_index%49]

        # Plot voxels
        ax.voxels(voxels, facecolors=colors, alpha=0.9, shade=False, linewidth=0.01)

        # =================== LOGGING =================== #

        # Save probability map
        probs_txt = np.vstack(packed_probs[:,container_index])
        np.savetxt(path + '-c' + str(container_index) + 'probs.txt', probs_txt, header = "Steps x Probabilities", newline="\n", delimiter = " ", fmt='%.8f')

        # Save the indices of the sampled boxes
        ptr_txt = np.vstack(packed_ptr[:,container_index])
        np.savetxt(path + '-c' + str(container_index) + 'ptr.txt', ptr_txt, header = "Steps x Chosen indices", newline="\n", delimiter = " ", fmt='%.8f')

        # Save Box matrices
        box_matrix_txt = np.vstack(packed_box_matrix[:,container_index])
        np.savetxt(path + '-c' + str(container_index) + 'box_matrix.txt', box_matrix_txt, header = "Static (steps x dims x boxes)", newline="\n", delimiter = " ", fmt='%.8f')

        # Save everything else (Rows, Columns, Bases, selected Boxes and Rewards) in a single text file
        cont_txt = np.vstack(packed_factors[:,:,container_index])

        holder_boxes = np.zeros((cont_txt.shape[0], 3))
        holder_cells = np.zeros((cont_txt.shape[0], 1))

        holder_boxes[:packed_boxes.shape[0]] = packed_boxes[:,container_index]
        big_txt = np.hstack((holder_boxes, cont_txt))

        holder_cells[:packed_base.shape[0],0] = packed_base[:, container_index]
        big_txt = np.hstack((holder_cells, big_txt))

        holder_cells[:packed_boxes.shape[0]] = packed_cols[:, container_index].squeeze(2)
        big_txt = np.hstack((holder_cells, big_txt))

        holder_cells[:packed_boxes.shape[0]] = packed_rows[:, container_index].squeeze(2)
        big_txt = np.hstack((holder_cells, big_txt))

        np.savetxt(path + '-c' + str(container_index) + '.txt', big_txt, header = "Row      Col         Base         Box                              Constr      Vol. Util.", newline="\n", delimiter = " ", fmt='%.8f')

        # Name the container figure with its final Volume Utilization reward
        fig_rew = V_rew[container_index].detach().numpy()
        # Save the figure
        matplotlib.pyplot.savefig(path + '-c' + str(container_index) + '-' + '%2.4f'%fig_rew + '.png', bbox_inches=0, dpi=400)

        matplotlib.pyplot.close()

    # if(kwargs['visual_demo']==True):

    #     choice=1
    #     print("Enter Container index for visual assmebly: ")
    #     input_index = int(input())
    #     while(choice):

    #         print("Chosen Container: ", input_index)

    #         visual_demo(packed_boxes[:,input_index], packed_rows[:,input_index], packed_cols[:,input_index], packed_base[:,input_index], packed_box_matrix[:,input_index], packed_probs[:,input_index], packed_ptr[:,input_index], kwargs['container_width'], kwargs['container_length'], kwargs['container_height'], x, y, z, box_colors, scale)

    #         print("Another? (If not, type in -1) ")
    #         input_index = int(input())

    #         if(input_index<0):
    #             break

    return

