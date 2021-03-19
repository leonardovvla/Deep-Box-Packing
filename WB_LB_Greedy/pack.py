"""

Wall Building LB-Greedy - Pack: Pack and Render containers through the WB-LB-Greedy algorithm

Author: Leonardo Albuquerque - ETH Zürich, 2021
Based on TAP-Net's original code from https://vcc.tech/research/2020/TAP

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tools

# ==================================================================================================== #
# =========================================== PACK AND RENDER ======================================== #
# ==================================================================================================== #

def pack_and_render(full_box_sequence, save_path, **kwargs):

    # Initial assignments
    dim1 = full_box_sequence.shape[1]
    dim2 = full_box_sequence.shape[2]

    block_dim = int((dim1 - 1))

    rotate_types = np.math.factorial(block_dim)

    blocks_num = int(dim2 / rotate_types)

    # Format boxes
    all_blocks = full_box_sequence.data[:,:1+block_dim,:].cpu().numpy()
    all_blocks = all_blocks.transpose(0, 2, 1).astype('int')

    # Get batch size
    sample_solution = full_box_sequence.data[:,:,:blocks_num]
    sample_solution = sample_solution.cpu().numpy()
    batch_size = sample_solution.shape[0]

    # Container configurations
    container_width  = kwargs['container_width']  * kwargs['unit']
    container_length  = kwargs['container_length']  * kwargs['unit']
    container_height = kwargs['container_height'] * kwargs['unit']

    container_width  = np.ceil(container_width).astype(int)
    container_length  = np.ceil(container_length).astype(int)
    container_height = np.ceil(container_height).astype(int)

    container_size = [container_width, container_length, container_height]
            
    # Initialize buffers
    my_ratio = []
    my_valid_size = []
    my_box_size = []
    my_empty_size = []
    my_stable_num = []
    my_packing_height = []

    # Set Heuristic choice
    calc_position_fn = tools.calc_positions_lb_greedy

    # For every sequence in the batch:
    for i in range(batch_size):

        print("Container ", i)

        plt.close('all')
        fig = plt.figure()

        # Full box sequence
        blocks = all_blocks[i]

        # Pack boxes according to WB-LB-Greedy
        positions, container, stable, ratio, scores = calc_position_fn(blocks, container_size, kwargs['iw'])
        valid_size, box_size, empty_num, stable_num, packing_height = scores

        # Log results
        my_ratio.append(ratio)
        my_valid_size.append(valid_size)
        my_box_size.append(box_size)
        my_empty_size.append(empty_num)
        my_stable_num.append(stable_num)
        my_packing_height.append(packing_height)

        # Draw containers
        tools.draw_container_voxel( container[:,:,:25], blocks_num, order=None,
            rotate_state=None,
            feasibility=None,
            view_type='back',
            save_name=save_path[:-4] + '-%d' % (i) + '-%2.4f' % (ratio))

    # Save information in the appropriate text files
    np.savetxt( save_path[:-13] + '-ratio.txt',             my_ratio)
    np.savetxt( save_path[:-13] + '-valid_size.txt',        my_valid_size)
    np.savetxt( save_path[:-13] + '-box_size.txt',          my_box_size)
    np.savetxt( save_path[:-13] + '-empty_size.txt',        my_empty_size)
    np.savetxt( save_path[:-13] + '-stable_num.txt',        my_stable_num)
    np.savetxt( save_path[:-13] + '-packing_height.txt',    my_packing_height)

    return np.mean(my_ratio), np.std(my_ratio)
