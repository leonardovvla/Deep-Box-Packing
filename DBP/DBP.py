"""

Deep Box Packing - Model: This file contains the full DBP structure and dynamics

Author: Leonardo Albuquerque - ETH Zürich, 2021
Based on TAP-Net's original code from https://vcc.tech/research/2020/TAP

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import transformer as tfm
import reward

from scipy import signal as sg
from operator import itemgetter

# ==================================================================================================== #
# ====================================== DEEP BOX PACKING CLASS ====================================== #
# ==================================================================================================== #

class DBP(nn.Module):

    def __init__(self, info_window_size, embed_size, forward_expansion, heads,
                use_cuda, container_width, container_length, container_height,
                heightmap_type, max_size, num_layers=6, dropout=0., unit=1, stable_placements=False):
        super(DBP, self).__init__()
        
        #General preparation and assignments
        heightmap_width = container_width * unit
        heightmap_length = container_length * unit
        heightmap_width = math.ceil(heightmap_width)
        heightmap_length = math.ceil(heightmap_length)  

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self.use_cuda = use_cuda

        self.container_width = math.ceil(container_width * unit)
        self.container_length = container_length
        self.container_height = container_height
        self.heightmap_type = heightmap_type

        self.max_size = max_size
        self.info_window_size = info_window_size
        self.stable_placements = stable_placements

        # Declaring Packer module (Wrapper for the Mapping Transformer)
        self.packer = tfm.Packer(box_max_size = max_size, container_max_size = container_height, 
                                 container_area = container_length*container_width, embed_size = embed_size, 
                                 num_layers = num_layers, forward_expansion = forward_expansion, heads = heads)

    # ========================================= AUXILIARY METHODS ========================================= #

    # This method builds the Border and Stability masks from the Broder and Stability dictionaries
    def get_masks(self, box_sequence, border_dicts, stab_dict, stable_placements):

        batch_size = box_sequence.shape[0]
        sequence_length = box_sequence.shape[2]

        # Prepare 6*n*W*L masks
        border_masks = np.ones((batch_size, sequence_length*self.container_width*self.container_length))
        stab_masks = np.ones((batch_size, sequence_length*self.container_width*self.container_length))

        w_sequence = box_sequence[:, 1, :].astype(int) #boxes' widths
        l_sequence = box_sequence[:, 2, :].astype(int) #boxes' lengths

        # Converting box bottom area into key indices for the stability dictionary
        stab_idx_seq = (self.max_size-1)*(w_sequence-1)+l_sequence-1 

        # For each box in each batch:
        for batch_idx in range(batch_size):
            for seq_idx in range(sequence_length):

                ### Border Mask computation

                # Check prohibitted positions given the box' width and length (so that the box will be fully contained)
                row_mask = border_dicts[0][w_sequence[batch_idx, seq_idx]]
                col_mask = border_dicts[1][l_sequence[batch_idx, seq_idx]]

                # Eliminate duplicated elements by taking the intercection of row_mask and col_mask
                box_mask = np.union1d(row_mask, col_mask).astype(int)
                # Convert the final elements into indices in the 6*n*W*L sequence
                box_mask = box_mask*sequence_length + seq_idx
                # Zero out prohibitted indices in the border mask
                border_masks[batch_idx, box_mask] = 0 

                ### Stability Mask computation

                # Only if it was specified that fully-stable placements should be enforced:
                if(stable_placements):

                    # Fetch stability map for this specific box in the Stability dictionary
                    # (Remember, the Stability dictionary is continuously updated in the UpdateRollout method)
                    box_stab_hm = stab_dict[stab_idx_seq[batch_idx, seq_idx]][batch_idx]

                    # Get the index for this specific box in every position of the 6*n*W*L vector
                    full_box_mask = np.arange(self.container_width*self.container_length)*sequence_length + seq_idx

                    # Mask the stability map for this box in the appropriate indices of the Stability mask
                    stab_masks[batch_idx, full_box_mask] = box_stab_hm

        return border_masks, stab_masks

    # This method looks up in the Height Dict the indices that have to be masked out so that no boxes can be packed beyond the ceiling of the container.
    def find_indices_all(self, height_dict, sat_box_matrix, hm_idx, sat_boxes):

        # Get widths and lengths of boxes which will saturate the height of the container if placed at hm_idx
        box_w = sat_box_matrix[1]
        box_l = sat_box_matrix[2]

        # Get rows and cols of the Heightmap which will saturate if boxes in sat_boxes are placed on top of
        hm_w = hm_idx[0]
        hm_l = hm_idx[1]

        # Get the flattened index of such positions in the heightmap
        index_in_hm = self.container_length*hm_w+hm_l

        # Get the indices of such box-position combinations in the Height Dictionary
        box_element = ((self.max_size-1)*(box_w-1)+box_l-1).reshape(-1,1)
        hm_element = (((self.max_size-1)*(self.max_size-1)*index_in_hm))

        box_element = np.tile(box_element,(1,index_in_hm.shape[0]))
        hm_element = np.tile(hm_element,(box_w.shape[0],1))

        dict_idx = (box_element+hm_element)

        # Here we want to return all elements of the 6*n*W*L vector which have to be masked out.
        # However, in this case we have reshaped the 6*n*W*L vector to be [6*n,W,L], in which
        # case, we want all [box,row,col] combinations which need to be masked out. These are 
        # represented here by [esb,w,l], which are lists holding the value of each box index, 
        # row and column that needs to be masked out in the [6*n,W,L] matrix (later rehaped back
        # to the standard 6*n*W*L format).

        w=[]
        l=[]
        esb=[] #expanded saturated boxes

        # For every key in the Height dictionary:
        for i in range(dict_idx.shape[0]):

            # Get the forbidden indices for each specific box_area-position combination and put them all in a list
            list_from_dict = [height_dict[d].astype(int) for d in dict_idx[i]]
            # Transform the list into an array
            array_from_dict = np.hstack(list_from_dict)
            # Separate rows and columns in their respective lists
            w.append(array_from_dict[0])
            l.append(array_from_dict[1])
            # Build the box index list (it is necessary to repeat box indeces so they match every position associated 
            # with them that needs to be masked, which is often more than one)
            esb.append(np.repeat(sat_boxes[i],array_from_dict[0].shape[0]))

        return w,l,esb

    # After every move made by DBP, this method updates the Height, Selection and Accessibility masks, the Stability dictionary and the Box matrix
    def updateRollout(self, selection_masks, height_masks, height_dict, stab_dict, access_masks, box_index, 
                      selection_dict, heightmap, box_matrix, full_box_sequence, conveyor_idx, fc_mask, access):

        batch_size = box_index.shape[0]
        blocks_num = full_box_sequence.shape[2]//6 #Total number of boxes to be packed

        height_masks = height_masks.reshape(batch_size, heightmap.shape[2], heightmap.shape[3], box_matrix.shape[2])

        # Select only the containers which are still being packed (fc_mask)
        hm = heightmap.squeeze().detach().numpy()[fc_mask]
        # The heightmap squared will serve for the stability mask computation
        hm_sq = hm*hm

        ### ============================================ Stability Dictionary Update ============================================ ###

        # The idea here is that a fully-stable position is one for which all elements in the heightmap which are coverd by the bottom
        # area of the box hold the same value. If this condition can be detected for every position in the heightmap that a given box
        # can occupy, then the problem of finding all stable positions for a specific box is solved. Here, we use an approximate
        # method to compute this condition with a high degree of certainty:

        # For every key (every possible box bottom area) in the Stability Dictionary:
        for k,v in stab_dict.items():

            # Create a custom heightmap with all positions initially declared unstable
            stab = np.zeros(hm.shape)

            w = int(k/(self.max_size-1) + 1) # Get the width of the box
            l = int(k%(self.max_size-1) + 1) # Get the length of the box

            # Build an all ones kernel from the dimensions. Ex: (w,l)=(1,2) ---> kernel = [1 1]
            # This kernel is nothing more than a matrix representation of the box
            kernel = np.ones((1,w,l))        

            # Here is the trick: in order to asses for each position if the elements under the kernel are the same, we perform a 
            # correlation between the kernel and the Heightmap. Because all kernel elements are 1, this is reduced to a simple average.
            # If the average of all elements under the kernel equals the value of an individual element, it is assumed there is a high 
            # chance that all elements are the same. In order to increase the certainty that this holds, the same procedure is done
            # with the hightmap squared: if the average of all elements squared also equals the value of an individual element squared,
            # there is an even higher chance that all elements are the same. Therefore:

            conv1 = sg.correlate(hm,kernel, mode='valid')/(w*l)     #Correlate the kernel with the heightmap and normalize
            conv2 = sg.correlate(hm_sq,kernel, mode='valid')/(w*l)  #Correlate the kernel with the heightmap squared and normalize

            mask1 = (hm[:,:conv1.shape[1],:conv1.shape[2]] == conv1).astype(int)     # Assess 1st order stability
            mask2 = (hm_sq[:,:conv2.shape[1],:conv2.shape[2]] == conv2).astype(int)  # Assess 2nd order stability

            # Cells which hold stable for both 1st and 2nd order stability for a given box are deemed stable placements for this box
            stab[:,:conv1.shape[1],:conv1.shape[2]] = mask1*mask2 # Combine the 1st and 2nd order assessments

            stab_dict[k][fc_mask] = stab.reshape(stab.shape[0],-1) # Update the Stability Mask

        ### ===================================================================================================================== ###

        ### Other Updates
        for batch_idx in range(batch_size):

            # It is only necessary to perform updates for containers which are still being packed:
            if(fc_mask[batch_idx]==True):

                #What was the box packed in this step
                box_in_batch = box_index[batch_idx]

                #What is now the current Heightmap in this step (already updated with box_in_batch)
                hm_in_batch = heightmap[batch_idx].squeeze().detach().numpy()

                ### ============================================= Accessibility Mask Update ============================================= ###

                # The idea is to iterate through the heightmap from the bottom row up and to assess,
                # for every column, what is the first row in which a box appears. As soon as the first
                # box appears in a column, it is considered blocked.

                # This vector tells for each column if it was already blocked
                access_open = np.ones(self.container_length)
                # This vector tells for each column the cummulative sum of its elements from the bottom up to the current row
                summed_rows = np.zeros(self.container_length)

                # Which row are we currenlty assessing in this iteration (start in the last row)
                slide_row = self.container_width-1
                # access map with inaccessible positions marked (start with every cell deemed inaccessible)
                access_hm = np.ones_like(hm_in_batch)

                # Only if it was specified that boxes can only be placed on accessible positions:
                if(access):

                    # While the top of the Heightmap has not been reached yet and there is at least one column in which no box has appeared:
                    while(slide_row>=0 and access_open.any()):

                        # Accumulate the values in the current row
                        summed_rows = hm_in_batch[slide_row]+summed_rows
                        # Check columns in which a box appeared
                        summed_mask = (summed_rows>0)

                        # Mark blocked columns
                        access_open[summed_mask] = 0
                        # For all columns which have not been found to be blocked, clear its access map in that row with a zero
                        access_hm[slide_row][~summed_mask] = 0

                        # Move on to the next row up
                        slide_row = slide_row-1

                    # Now remember that the first inaccessible positions in the access map are still deemed accessible because
                    # the first blocking boxes can usually be temporarily lifted for another box to be placed behind them. In this
                    # sense, we now "carve" one inaccessible layer out of the access map. We do that by building an extended version
                    # of the map and then subsequently carving it from the bottom up and then from each unblocked side.

                    ext_access_hm = np.zeros((self.container_width+2, self.container_length+4))
                    ext_access_hm[:self.container_width, 2:self.container_length+2] = access_hm
                    ext_access_hm[:self.container_width,0] = access_hm[:,0]
                    ext_access_hm[:self.container_width,1] = access_hm[:,0]
                    ext_access_hm[:self.container_width,-1] = access_hm[:,-1]
                    ext_access_hm[:self.container_width,-2] = access_hm[:,-1]

                    for i in range(1,3):

                        access_hm_left_mask = ext_access_hm[:self.container_width,(2+i):self.container_length+(2+i)]
                        access_hm_right_mask = ext_access_hm[:self.container_width,(2-i):self.container_length+(2-i)]
                        access_hm_up_mask = ext_access_hm[i:self.container_width+i,2:self.container_length+2]
     
                        access_hm = access_hm*access_hm_left_mask*access_hm_right_mask*access_hm_up_mask

                    access_masks[batch_idx] = np.repeat(np.logical_not(access_hm).flatten(),box_matrix.shape[2])

                ### ===================================================================================================================== ###

                if(conveyor_idx>=blocks_num):
                    ### =================== Selection Mask Update =================== ###
                    selection_masks[batch_idx, selection_dict[box_in_batch]] = 0
                    ### ============================================================= ###

                else:
                    ### ===================== Box matrix Update ============================================================================== ###
                    box_matrix[batch_idx,1:,box_matrix[batch_idx,0]==box_in_batch] = full_box_sequence[batch_idx,1:,full_box_sequence[batch_idx,0]==conveyor_idx]
                    ### ====================================================================================================================== ###

                    # While the Box matrix is still receiving new boxes at every move, always recompute the height mask from scratch
                    # (This can definitely be optimized)
                    height_masks[batch_idx] = np.ones_like(height_masks[batch_idx])

                ### ================================================= Height Mask Update ================================================= ###

                # Check all heights of boxes still available for packing in this batch
                available_heights = box_matrix[batch_idx,3,:].detach().numpy()*selection_masks[batch_idx,:box_matrix.shape[2]]
                # List them in decreasing order
                tallest_boxes = np.flip(np.unique(available_heights),0)

                #We are going to start from the tallest box and iterate down the list.
                index = 0

                # Check all positions in the Heightmap which, summed with the current height being analyzed, surpasses the container height
                saturated_cells = (tallest_boxes[index]+hm_in_batch > self.container_height)

                # If there are any cells that saturate for this hight, we need to update the mask. 
                # If there are no cells that saturate for this height, certainly there are no cells that saturate for shroter heights.
                # Therefore, once a given height does not saturate any cell anymore, we don't need to keep analyzing other heights down the list.
                while(saturated_cells.any()):

                    # Check the indices of the boxes from the box matrix which have the current height being analyzed
                    saturated_boxes = (box_matrix[batch_idx,3,:] == tallest_boxes[index]).nonzero().squeeze()
                    # Get the indices of the saturaced cells of the Heightmap
                    saturated_idx = np.vstack((saturated_cells.nonzero()[0], saturated_cells.nonzero()[1]))

                    # Look up in the Height Dict the indices that have to be masked out so that the saturated boxes cannot be on top of the saturated cells
                    w,l,exp_sat_boxes = self.find_indices_all(height_dict,box_matrix[batch_idx,:,saturated_boxes].detach().numpy(),saturated_idx, saturated_boxes.detach().numpy())

                    # Update the Height mask
                    for i in range(len(w)):
                        height_masks[batch_idx,w[i],l[i],exp_sat_boxes[i]]=0

                    # Move on to the next available height
                    index = index+1
                    # If all heights have been analyzed, break
                    if(index >= tallest_boxes.shape[0]):
                        break

                    # Check all positions in the Heightmap which, summed with the new height being analyzed, surpasses the container height
                    saturated_cells = (tallest_boxes[index]+hm_in_batch > self.container_height)

                #### ====================================================================================================================== ###

        # Convert Height mask back to the 6*n*W*L shape.
        height_masks = height_masks.reshape(batch_size, heightmap.shape[2]*heightmap.shape[3]*box_matrix.shape[2])

        # Update the slide index of the conveyor belt (tells which is the next box to be inserted in the Information window)
        conveyor_idx=conveyor_idx+1

        return [selection_masks, height_masks, stab_dict, access_masks, box_matrix, conveyor_idx]


    # ========================================= FORWARD METHOD ========================================= #    

    def forward(self, full_box_sequence, heightmap, mask_dicts, stab_dict, access, draw):

        # In 3D, every box has 6 different possible orientations
        rotate_types = 6

        batch_size, box_dims, sequence_size = full_box_sequence.size()

        #Size of the box_matrix
        vis_sequence_size = self.info_window_size*rotate_types

        # Initialize the Reward class
        rew = reward.Reward(batch_size=batch_size, container_width=self.container_width, container_length=self.container_length, container_height=self.container_height)

        # Total number of boxes in the conveyor belt
        blocks_num = int(sequence_size / rotate_types)

        #Dimensions of the container
        container_size = [self.container_width, self.container_length, self.container_height]

        # Structures for holding the output sequences
        tour_idx, tour_logp, pred_vals, scores, = [], [], [], []

        # The maximum number of packing steps if obviously the number of boxes in the conveyor belt
        max_steps = blocks_num

        # Initializing information buffer for the visualization process
        draw_info = None

        # Only called in Validation
        if(draw):

            # Initialize all information buffers that are needed for posteriorly visualizing the packing process step-by-step.
            # packed_boxes: all boxes packed at each step
            # packed_rows: all rows of the Heightmap selected at each step
            # packed_cols: all columns of the Heightmap selected at each step
            # packed_box_matrix: which boxes DBP chose from at each step 
            # packed_probs: the action probability vector form which DBP sampled at each step
            # packed_ptr: index of the sampled boxes in the 6*n*W*L vector
            # packed_base: the height at which each box was packed at each step
            # packed_factors: Rewards obtained at each step formatted for text file logging

            packed_boxes, packed_rows, packed_cols, packed_box_matrix, packed_probs, packed_ptr = [], [], [], [], [], []
            packed_base = np.zeros((blocks_num, batch_size))
            packed_factors = np.zeros((blocks_num+1,2,batch_size))

        # Initialize Selection, Height and Accessibility masks, as well as Selection and Height dictionaries
        selection_masks = np.ones((batch_size, vis_sequence_size*heightmap.shape[2]*heightmap.shape[3]))
        height_masks = np.ones((batch_size, vis_sequence_size*heightmap.shape[2]*heightmap.shape[3]))
        access_masks = np.ones((batch_size, vis_sequence_size*heightmap.shape[2]*heightmap.shape[3]))

        selection_dict = mask_dicts[2]
        height_dict = mask_dicts[3]

        # Initialize the Box matrix with the first 6*n elements of the full_box_sequence
        box_matrix = full_box_sequence[:,:,full_box_sequence[0,0]<self.info_window_size]

        # Initialize the slide index of the conveyor belt, which tells the next box to be inserted in the Information window
        conveyor_idx = self.info_window_size

        # Initialize the mask that tells which containers are already full and don't need to be packed anymore
        full_containers_mask = np.ones(batch_size).astype(bool)


        # =============== PACKING LOOP =============== #

        for _ in range(max_steps):

            # If there are no more boxes to pack, end loop
            if not selection_masks.any():
                break

            # Build the Border and Stability masks
            border_masks, stab_masks = self.get_masks(box_matrix.clone().detach().numpy(), mask_dicts[0:2], stab_dict, self.stable_placements)

            # Combine all DBP constraint masks
            union_masks = (border_masks * selection_masks * height_masks * stab_masks * access_masks)
            full_mask_log = (torch.tensor(border_masks).log() + torch.tensor(selection_masks).log() + torch.tensor(height_masks).log() + torch.tensor(stab_masks).log() + torch.tensor(access_masks).log())

            # Check if constraint masks are deeming any container as full. If the sum of the combined masks is zero for a given
            # container, that means the constraint masks will not allow boxes to be packed in this container anymore.
            pack_complete = np.sum(union_masks, axis=1)
            if not pack_complete.all():

                # Update full containers maks
                full_containers_mask[pack_complete==0] = 0

                # If all containers are full:
                if not pack_complete.any():
                    if(draw):
                        print("    All Validation Containers full at box ", conveyor_idx-self.info_window_size)
                    break

            # Indices of the containers still being packed (not classified as full)
            ongoing_containers = np.nonzero(full_containers_mask)[0]

            # Pass the Box matrix and the Heightmap into the Mapping Transformer.
            # Outputs are the action probability vector and the value estimated for the current state.
            # Note that these are actually still "raw logits", since they haven't been softmaxed yet.
            probs, vals = self.packer(box_matrix, heightmap)

            # Wrapper for the action probability vectors which replaces the probabilities for '-1' 
            # in the containers which have already been considered full. These will serve only to 
            # signal that no sample is going to be drawn from these vectors.
            masked_probs = torch.zeros(ongoing_containers.shape[0], probs.shape[1])-1
            for c in range(masked_probs.shape[0]):
                # Apply the constraint masks to the action probability vector.
                masked_probs[c] = (probs + full_mask_log)[ongoing_containers[c]]

            # After applying the masks, Softmax it to get actual probabilities.
            probs = F.softmax(masked_probs, dim=1)

            # Initialize wrappers for the sampled indices and for the log probabilities of the sampled actions.
            # These are necessary since some containers will not have any action sampled. These elements will 
            # keep holding the wrapper standard values (0 for ptr and -inf for logp).
            ptr = (torch.zeros(batch_size)-1).type(torch.LongTensor)
            logp = torch.zeros(batch_size)-float('inf')

            # If training, sample from the Action probability vector
            if self.training:
                m = torch.distributions.Categorical(probs)
                part_ptr = m.sample()
                log_prob = m.log_prob(part_ptr)

            # If testing (validating), choose greedy action
            else:
                prob, part_ptr = torch.max(probs, 1)
                log_prob = prob.log()

            # Insert sampled values in the ptr and logp wrappers
            i=0
            for c in ongoing_containers:
                logp[c] = log_prob[i]
                i=i+1

            ptr[full_containers_mask] = part_ptr

            v_ptr = part_ptr.view(-1, 1, 1).detach().numpy()

            # Get the index of the sampled box for every container in the batch (index in the 6*n*W*L vector)
            box_index = (np.zeros(batch_size)-1).astype(int)
            box_index[full_containers_mask] = np.remainder(np.remainder(v_ptr.flatten(),vis_sequence_size),(vis_sequence_size/6)).squeeze()
            # Get the row of the sampled placement for every container in the batch
            line_of_box = (np.zeros((batch_size,1,1))-1).astype(int)
            line_of_box[full_containers_mask] = ((v_ptr//vis_sequence_size)//heightmap.shape[3])
            # Get the column of the sampled placement for every container in the batch
            column_of_box = (np.zeros((batch_size,1,1))-1).astype(int)
            column_of_box[full_containers_mask] = ((v_ptr//vis_sequence_size)%heightmap.shape[3])

            # Find which box in the Box matrix was sampled
            box_idx_in_box_matrix = (v_ptr%box_matrix.shape[2]).squeeze()

            # Get the dimensions of the sampled boxes
            sampled_boxes = np.zeros((batch_size,3))
            sampled_boxes[full_containers_mask] = box_matrix[full_containers_mask,1:,box_idx_in_box_matrix].detach().numpy()
            blocks = sampled_boxes.astype(int) 

            # If in Validation, collect all information relevant to the step-by-step visualization in the respective buffers
            if(draw):
                packed_boxes.append(blocks)
                packed_rows.append(line_of_box)
                packed_cols.append(column_of_box)
                packed_box_matrix.append(box_matrix.clone().detach().numpy())

                packed_probs_mask = np.zeros((batch_size, vis_sequence_size*heightmap.shape[2]*heightmap.shape[3]))
                packed_probs_mask[full_containers_mask] = probs.clone().detach().numpy()

                packed_probs.append(packed_probs_mask)
                packed_ptr.append(ptr.clone().detach().numpy())
            
            ### ===================================== HIGHTMAP UPDATE ===================================== ###

            for batch_index in range(batch_size):

                # Only if this container is still being packed:
                if(full_containers_mask[batch_index]==True):

                    # Get the indeces of every cell that the height of the sampled box will be added to
                    line_index = ((np.arange(blocks[batch_index][0])).repeat(blocks[batch_index][1])+line_of_box[batch_index]).squeeze()
                    column_index = (np.tile(np.arange(blocks[batch_index][1]),blocks[batch_index][0])+column_of_box[batch_index]).squeeze()

                    # Get the maximum element in the Heightmap spanned by the bottom area of the box
                    hm_below_box = heightmap[batch_index][0][line_index,column_index].detach().numpy()
                    max_height_below_box = np.max(hm_below_box)
                    
                    # If in validation, add the max height information to its respective buffer for posterior reconstruction
                    if(draw): 
                        packed_base[len(packed_boxes)-1, batch_index] = max_height_below_box

                    # Calculate the new height and add it to the hightmap in the appropriate cells where the new box is positioned
                    new_height = blocks[batch_index][2] + max_height_below_box
                    heightmap[batch_index][0][line_index,column_index] = new_height

            ### =========================================================================================== ###

            #Update the Selection, Height and Accessibility masks, the Stability dictionary, the Box matrix and the coonveyor index
            [selection_masks, height_masks, stab_dict, access_masks, box_matrix, conveyor_idx] = self.updateRollout(selection_masks, height_masks, height_dict, stab_dict, access_masks, box_index, selection_dict, heightmap, box_matrix, full_box_sequence, conveyor_idx, full_containers_mask, access)

            # Compute local Constructive reward
            rewards, factors = rew.calc_roll_reward(line_of_box, sampled_boxes, full_containers_mask, False)

            # Experience buffer update
            pred_vals.append(vals.unsqueeze(1))
            tour_logp.append(logp.unsqueeze(1))
            scores.append(rewards.unsqueeze(1))

            # If in validation, collect the logging data for the text files
            if(draw):
                packed_factors[len(packed_boxes)-1, :, :] = factors

        # Final experience buffer formatting
        pred_vals = torch.cat(pred_vals, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)
        scores = torch.cat(scores, dim=1)

        # Compute final Volume Utilization reward for all containers
        V_rew, factors = rew.calc_roll_reward(line_of_box, sampled_boxes, full_containers_mask, True)

        # Compute compound reward (average of constructive terms and final volume utilization)
        for i in range(V_rew.shape[0]):
            scores[i,:] = (scores[i,:]+V_rew[i])/2

        if self.use_cuda:
                scores = scores.cuda()

        space = ''

        # If in validation, gather all the information for visualization processes
        if(draw):
            packed_factors[len(packed_boxes), :, :] = factors
            draw_info = [packed_boxes, packed_rows, packed_cols, packed_base, packed_box_matrix, packed_probs, packed_ptr, packed_factors]
            space = '    '

        return tour_logp, -scores, pred_vals, V_rew, draw_info
