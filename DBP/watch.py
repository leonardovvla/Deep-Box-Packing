"""

Deep Box Packing - Watch: Call to step-by-step visualization of DBP packing procedure

Author: Leonardo Albuquerque - ETH Zürich, 2021

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import numpy as np
from draw import visual_demo

# ==================================================================================================== #
# ========================================== VISUAL DEMO CALL ======================================== #
# ==================================================================================================== #

if __name__ == "__main__":

    # Specify the path to the container that will be iteratively packed (./results/<iw>/<exp_name>/render/<epoch_name>/batch<_>-c<_>)
    main_path = './results/10/DBP-Thesis-Test-Stable_True-testing-150-320-32-32-2-1-1-Constructive-Utilization-0.0005-2021-03-18-12-11/render/0__0.9550__0.0238/batch0-c28'

    # Path definition to all text files that contain relevant information for this visualization
    path_draw_info = main_path + '.txt'
    path_probs = main_path + 'probs.txt'
    path_ptr = main_path + 'ptr.txt'
    path_box_matrix = main_path + 'box_matrix.txt'

    # Load data from text files
    data = np.loadtxt(path_draw_info).astype('float32')
    packed_probs = np.loadtxt(path_probs).astype('float32')
    packed_ptr = np.loadtxt(path_ptr).astype('float32')
    packed_box_matrix = np.loadtxt(path_box_matrix).astype('float32')

    packed_box_matrix = packed_box_matrix.reshape(packed_ptr.shape[0],4,packed_box_matrix.shape[1])

    packed_rows = data[:,0]
    packed_cols = data[:,1]
    packed_base = data[:,2]
    packed_boxes = data[:,3:6]

    # Define container boundaries
    c_w = 8
    c_l = 6
    c_h = 6
    x, y, z = np.indices((c_l, c_w, c_h))

    # Palette of colors for coloring boxes
    box_colors = ['#0b55b5', '#ffa600', '#4bab3c', '#e0e048', '#cc2f2f', '#50bdc9', 
                  '#a47c5f', '#321f91', '#edecf9', '#dd9286',  '#91aec6', 'silver',
                  'khaki', '#ada5de', 'coral', '#ffd64d', '#6e6702', 'yellowgreen',
                  'lightblue', 'salmon', '#004fa4', 'tan', '#fdff04', 'lavender', '#037272', 
                  '#eb1629', '#2a3132', '#336b87', '#f27657', '#598234', '#e53a3a', '#aebd38', '#c4dfe6',
                  '#44b8b1', '#ffbb00', '#fb6542', '#375e97', '#4cb5f5', '#f4cc70', '#1e434c', '#9b4f0f',
                  '#f1f1f2', '#bcbabe', '#20c0df', '#011a27', '#eed8ae', '#f0810f', '#4b7447', '#2d4262']

    # Projection scale under three axes
    y_scale=1.0
    x_scale=c_l/c_w
    z_scale=c_h/c_w

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    # Call visual demo
    visual_demo(packed_boxes, packed_rows, packed_cols, packed_base, packed_box_matrix, packed_probs, packed_ptr, c_w, c_l, c_h, x, y, z, box_colors, scale)

