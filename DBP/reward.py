"""

Deep Box Packing - Reward Function

Author: Leonardo Albuquerque - ETH Zürich, 2021

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import torch
import numpy as np

# ==================================================================================================== #
# ============================================ REWARD CLASS ========================================== #
# ==================================================================================================== #

class Reward():
    def __init__(self, batch_size, container_width, container_length, container_height):

        super(Reward, self).__init__()

        self.summed_box_volume = torch.zeros(batch_size)

        self.beta = 1.0     # Optional weight for Constructive Reward
        self.omega = 1.0    # Optional weight for Volume Utilization Reward

        self.batch_size = batch_size
        self.container_width = container_width
        self.container_volume = container_width*container_length*container_height

        self.envelope_row_line = np.zeros(batch_size)

    def calc_roll_reward(self, row, box, fc_mask, end):

        # Accumulate packed box volume per container
        box_volume = np.prod(box, axis=1)
        self.summed_box_volume = self.summed_box_volume + box_volume

        # Local Constructive Reward
        if(end==False):

            # ====== Deprecated Constructive reward functions ====== #
            # back = torch.tensor(np.exp(-row).squeeze())
            # back = (1.0 - torch.true_divide(torch.tensor(row),self.container_width-1)).squeeze()

            # ======== Current Constructive reward function ======== #

            # Frontal limit of positioned box
            line_diff = row.squeeze()+box[:,0]-1
            # Mask telling which placements were made properly (ahead or equal to the front line)
            proper_back = (line_diff >= self.envelope_row_line).astype(bool)
            # Placements receive a 1 reward if made properly, 0 otherwise
            back = torch.ones(self.batch_size)
            back[~proper_back] = 0
            # Update container front line with new placement if applicable
            self.envelope_row_line[proper_back] = line_diff[proper_back]

            # Apply weight
            reward = self.beta*back

            # Containers which are already full always receive 0 additional reward for the round
            not_fc_mask = ~fc_mask
            reward[not_fc_mask] = 0

            # Dummy volume utilization reward for logging purposes (not used for optimization)
            V_rew = torch.zeros_like(back)
            # Data for logging in text file
            factors = np.vstack((back,V_rew))

        # Global Volume Utilization Reward
        if(end==True):

            # Compute Volume Utilization
            V_rew = torch.exp(-torch.true_divide((self.container_volume - self.summed_box_volume),self.container_volume).squeeze())
            # Apply weight
            reward = self.omega*V_rew

            # dummy constructive reward for logging purposes (not used for optimization)
            back = torch.zeros_like(V_rew)
            # Data for logging in text file
            factors = np.vstack((back,V_rew))

        return reward.squeeze(), factors




# ========================== Deprecated Reward function ========================== #
#                                                                                  #
#    Even though they were not used, several (potentially useful) local reward     #
#        computations were left here for the convenice of future developers.       #
#              Descriptions for these computations are given below.                #
#                                                                                  #
# ================================================================================ #

    def calc_reward(self, hm, prev_hm, row, col, box, max_h, hm_below_box, fc_mask):

        # hm: updated Heightmap
        # prev_hm: Heightmap beofre update
        # row: row of packing
        # col: column of packing
        # box: box packed
        # max_h: height at which the box was packed
        # hm_below_box: Heightmap cells underneath packed box
        # fc_mask: full_containers_mask

        hm = hm.reshape(hm.shape[0], hm.shape[2], hm.shape[3])
        row = row.squeeze()
        col = col.squeeze()

        #Cummulative values
        box_volume = torch.prod(box, axis=1)
        self.summed_box_volume = self.summed_box_volume + box_volume

        box_area = 2*box[:,0]*box[:,1] + 2*box[:,1]*box[:,2] + 2*box[:,0]*box[:,2]
        self.summed_box_area = self.summed_box_area + box_area

        # ============= Envelope Computation (min. parallelogram which encompasses all packed boxes) ============= #

        row_back_indeces = ((row < self.envelope_back_corner[:,0])*fc_mask).type(torch.bool)
        col_back_indeces = ((col < self.envelope_back_corner[:,1])*fc_mask).type(torch.bool)
        row_front_indeces = ((row + box[:,0] > self.envelope_front_corner[:,0])*fc_mask).type(torch.bool)
        col_front_indeces = ((col + box[:,1] > self.envelope_front_corner[:,1])*fc_mask).type(torch.bool)

        self.envelope_back_corner[row_back_indeces,0] = row[row_back_indeces].float()
        self.envelope_back_corner[col_back_indeces,1] = col[col_back_indeces].float()
        self.envelope_front_corner[row_front_indeces,0] = row[row_front_indeces] + box[row_front_indeces,0]
        self.envelope_front_corner[col_front_indeces,1] = col[col_front_indeces] + box[col_front_indeces,1]
        self.envelope_height, _ = torch.max(hm.reshape(hm.shape[0],hm.shape[1]*hm.shape[2]),1)

        envelope_sides = self.envelope_front_corner-self.envelope_back_corner
        envelope_volume = envelope_sides[:,0]*envelope_sides[:,1]*self.envelope_height

        # ======================================================================================================== #

        for batch_index in range(self.batch_size):

            line_index = ((np.arange(box[batch_index][0])).repeat(box[batch_index][1])+row[batch_index]).squeeze()
            column_index = (np.tile(np.arange(box[batch_index][1]),box[batch_index][0])+col[batch_index]).squeeze()

            #Area below box which is not in contact with its bottom (only !=0 if the placement is not fully stable)
            exposed_under_area = torch.sum(hm_below_box != max_h)
            #Area below box which is in contact with its bottom 
            touch_under_area =  torch.sum(hm_below_box == max_h)
            #Area below box which is in contact with its bottom and that is not the floor (belongs to other boxes)
            touch_box_under_area =  torch.sum((hm_below_box == max_h)*(hm_below_box != 0.0))

            # ===================================== Stability Reward computation ==================================== #

            stability_sparse[batch_index] = (exposed_under_area==0).type(torch.LongTensor)

            stability_dense[batch_index] = torch.true_divide(touch_under_area, (touch_under_area+exposed_under_area))

            # ============ Exposed Area computation (Total exposed surface area of the packed boxes) ================ #

            new_height = box[batch_index][2] + max_h

            # Generate an empty container called "lone_map"
            lone_map = torch.zeros(prev_hm[batch_index][0].shape).type(torch.LongTensor)
            # Place the box in it just as if it were to be placed in the actual container
            lone_map[line_index,column_index] = box[batch_index][2]

            # Compute the top area of the box (unless it is in contact with the ceiling, then it is not exposed)
            lone_top_area = box[batch_index,0]*box[batch_index,1]*(new_height!=self.container_height)
            # Compute the side area of the box (note that if it is in contact with the left of right wall, it is not going to be counted)
            lone_side_map = lone_map[:,:-1]-lone_map[:,1:]
            lone_side_area = torch.sum(torch.abs(lone_side_map))
            # Compute the frontal/ back area of the box (note that if it is in contact with the back wall, it is not going to be counted)
            lone_frontal_map = lone_map[:-1,:]-lone_map[1:,:]
            lone_frontal_area = torch.sum(torch.abs(lone_frontal_map)) + torch.sum(lone_map[-1:,:])
            # Total sum of exposed box area considering it is in an empty container. This is called the "lone area"
            lone_area = lone_top_area + lone_side_area + lone_frontal_area

            # Locate the cells that are on the contour of the box
            lone_side_mask = (lone_side_map!=0).type(torch.LongTensor)
            lone_frontal_mask = (lone_frontal_map!=0).type(torch.LongTensor)

            # Take a horizontal section of the container (before having packed the box) delimited below by the height   
            # on which the new box is going to be packed (the "floor" of the box) and above by the new box' height.
            touch_sides_map = torch.clamp((prev_hm[batch_index][0]-max_h),0,box[batch_index][2])
            # Calculate the area of all previously packed boxes which will be in lateral touch with the new box.
            touch_sides_side = torch.sum(torch.abs(touch_sides_map[:,:-1]-touch_sides_map[:,1:])*lone_side_mask)
            touch_sides_frontal = torch.sum(torch.abs(touch_sides_map[:-1,:]-touch_sides_map[1:,:])*lone_frontal_mask)
            # Total sum of "touch" areas. In other words, this is the lateral area the new box is going to cover form other boxes.
            touch_sides_area = touch_sides_side + touch_sides_frontal

            # The total area a new box adds to the surface area of a packing
            added_area[batch_index] = lone_area + exposed_under_area - touch_box_under_area - touch_sides_area

            # Cummulative exposed area of a packing
            self.envelope_exposed_area = self.envelope_exposed_area + added_area

