
"""

Wall Building LB-Greedy - Tools: WB-LB-Greedy algorithm and auxiliary visualization functions

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
from mpl_toolkits.mplot3d import Axes3D
import itertools
import copy
import time

###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
###patch end###

# ==================================================================================================== #
# ===================================== WALL BUILDING LB-GREEDY ====================================== #
# ==================================================================================================== #

# Left-Bottom corners by greedy EMS strategy (TAP-Net implementation)
def calc_one_position_lb_greedy_3d(block, block_index, container_size,
                                container, positions, stable, heightmap, valid_size, empty_size):
    """
    calculate the latest block's position in the container by lb-greedy in 2D cases
    ---
    params:
    ---
        static params:
            block: int * 3 array, size of the block to pack
            block_index: int, index of the block to pack, previous were already packed
            container_size: 1 x 3 array, size of the container

        dynamic params:
            container: width * length * height array, the container state
            positions: int * 3 array, coordinates of the blocks, [0, 0] for blocks after block_index
            stable: n * 1 bool list, the blocks' stability state
            heightmap: width * length array, heightmap of the container
            valid_size: int, sum of the packed blocks' size
            empty_size: int, size of the empty space under packed blocks
    return:
    ---
        container: width * length * height array, updated container
        positions: int * 3 array, updated positions
        stable: n * 1 bool list, updated stable
        heightmap: width * length array, updated heightmap
        valid_size: int, updated valid_size
        empty_size: int, updated empty_size
    """
    placed=False

    block_dim = len(block)
    block_x, block_y, block_z = block
    valid_size += block_x * block_y * block_z

    # get empty-maximal-spaces list from heightmap
    # each ems represented as a left-bottom corner
    ems_list = []
    # hm_diff: height differences of neightbor columns, padding 0 in the front
    # x coordinate
    hm_diff_x = np.insert(heightmap, 0, heightmap[0, :], axis=0)
    hm_diff_x = np.delete(hm_diff_x, len(hm_diff_x)-1, axis=0)
    hm_diff_x = heightmap - hm_diff_x
    # y coordinate
    hm_diff_y = np.insert(heightmap, 0, heightmap[:, 0], axis=1)
    hm_diff_y = np.delete(hm_diff_y, len(hm_diff_y.T)-1, axis=1)
    hm_diff_y = heightmap - hm_diff_y

    # get the xy coordinates of all left-deep-bottom corners
    ems_x_list = np.array(np.nonzero(hm_diff_x)).T.tolist()
    ems_y_list = np.array(np.nonzero(hm_diff_y)).T.tolist()
    ems_xy_list = []
    ems_xy_list.append([0,0])
    for xy in ems_x_list:
        x, y = xy
        if y!=0 and [x, y-1] in ems_x_list:
            if heightmap[x, y] == heightmap[x, y-1] and \
                hm_diff_x[x, y] == hm_diff_x[x, y-1]:
                continue
        ems_xy_list.append(xy)
    for xy in ems_y_list:
        x, y = xy
        if x!=0 and [x-1, y] in ems_y_list:
            if heightmap[x, y] == heightmap[x-1, y] and \
                hm_diff_x[x, y] == hm_diff_x[x-1, y]:
                continue
        if xy not in ems_xy_list:
            ems_xy_list.append(xy)

    # sort by y coordinate, then x
    def y_first(pos): return pos[1]
    ems_xy_list.sort(key=y_first, reverse=False)

    # get ems_list
    for xy in ems_xy_list:
        x, y = xy
        if x+block_x > container_size[0] or \
            y+block_y > container_size[1]: continue
        z = np.max( heightmap[x:x+block_x, y:y+block_y] )
        if(z<container_size[2]):
            ems_list.append( [ x, y, z ] )
    
    # firt consider the most bottom, sort by z coordinate, then y last x
    def z_first(pos): return pos[2]
    ems_list.sort(key=z_first, reverse=False)

    # if no ems found
    if len(ems_list) == 0:
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        return container, positions, stable, heightmap, valid_size, empty_size, placed

    # varients to store results of searching ems corners
    ems_num = len(ems_list)
    pos_ems = np.zeros((ems_num, block_dim)).astype(int)
    is_settle_ems  = [False] * ems_num
    is_stable_ems  = [False] * ems_num
    compactness_ems  = [0.0] * ems_num
    pyramidality_ems = [0.0] * ems_num
    stability_ems    = [0.0] * ems_num
    empty_ems = [empty_size] * ems_num
    under_space_mask  = [[]] * ems_num
    heightmap_ems = [np.zeros(container_size[:-1]).astype(int)] * ems_num
    visited = []

    # Remember that here the container is discretized by an actual 3D voxel grid

    # check if a position suitable by analyzing overlap, stability and containment constraints
    def check_position(index, _x, _y, _z):
        # check if the pos visited
        if [_x, _y, _z] in visited: 
            return
        # If not fully-stable:
        if _z>0 and (container[_x:_x+block_x, _y:_y+block_y, _z-1]==0).any(): 
            # Packing is not allowed (Stability constraint enforcement)
            return
        visited.append([_x, _y, _z])
        # If the voxels this box would occupy are all free:
        if (container[_x:_x+block_x, _y:_y+block_y, _z] == 0).all():
            # It is a stable placement (because if it wasn't it would have returned before)
            is_stable_ems[index] = True
            pos_ems[index] = np.array([_x, _y, _z])

            # If the sum of this box' height with the max height packed underneath it does not surpass the max height of the container:
            if((_z + block_z)<=container_size[2]):
                # Add Box to the voxel grid
                heightmap_ems[index][_x:_x+block_x, _y:_y+block_y] = _z + block_z
                # Successfull placement
                is_settle_ems[index] = True
            else:
                # Packing is not allowed (Height Containment constraint enforcement)
                return

        return

    # calculate reward
    def calc_C(index):
        _x, _y, _z = pos_ems[index]

        # compactness
        front_line_ems = np.max(np.nonzero(heightmap_ems[index])[0])+1
        bbox_size = front_line_ems * container_size[1] * container_size[2]
        # Sum of packed box volume / volume of shortest possible container in the depth dimension that could contain the packed boxes
        compactness_ems[index] = valid_size / bbox_size

    # search positions in each ems
    X = int(container_size[0] - block_x + 1)
    Y = int(container_size[1] - block_y + 1)
    for ems_index, ems in enumerate(ems_list):
        # using bottom-left strategy in each ems
        heightmap_ems[ems_index] = heightmap.copy()
        X0, Y0, _z = ems
        for _x, _y  in itertools.product( range(X0, X), range(Y0, Y) ):
            if is_settle_ems[ems_index]: break
            check_position(ems_index, _x, _y, _z)
        if is_settle_ems[ems_index]: 
            calc_C(ems_index)
            placed=True

    # if the block has not been settled
    if np.sum(is_settle_ems) == 0:
        # Eliminate it from the accumulated packed box volume
        valid_size -= block_x * block_y * block_z
        stable[block_index] = False
        # Return
        return container, positions, stable, heightmap, valid_size, empty_size, placed

    # If the block was packed, get its best ems
    ratio_ems = compactness_ems
    best_ems_index = np.argmax(ratio_ems)
    while not is_settle_ems[best_ems_index]:
        ratio_ems.remove(ratio_ems[best_ems_index])
        best_ems_index = np.argmax(ratio_ems)

    # update the dynamic parameters
    _x, _y, _z = pos_ems[best_ems_index]
    container[_x:_x+block_x, _y:_y+block_y, _z:_z+block_z] = block_index + 1
    container[_x:_x+block_x, _y:_y+block_y, 0:_z][ under_space_mask[best_ems_index] ] = -1
    positions[block_index] = pos_ems[best_ems_index]
    stable[block_index] = is_stable_ems[best_ems_index]
    heightmap = heightmap_ems[best_ems_index]
    empty_size = empty_ems[best_ems_index]

    return container, positions, stable, heightmap, valid_size, empty_size, placed

def calc_positions_lb_greedy(blocks, container_size, num_nodes):
    '''
    calculate the positions to pack a group of blocks into a container by lb-greedy
    ---
    params:
    ---
        blocks: n x 2/3 array, blocks with an order
        container_size: 1 x 2/3 array, size of the container

    return:
    ---
        positions: int x 2/3 array, packing positions of the blocks
        container: width (* depth) * height array, the final state of the container
        stable: n x 1 bool list, each element indicates whether a block is placed(hard)/stable(soft) or not
        ratio: float, C / C*S / C+P / (C+P)*S / C+P+S, calculated by the following scores
        scores: 5 integer numbers: valid_size, box_size, empty_size, stable_num and packing_height
    '''

    lb_start = time.time()

    # Initialize actual environment variables
    blocks = blocks.astype('int')
    blocks_num = int(len(blocks)/6)
    block_dim = len(blocks[0])-1
    positions = np.zeros((blocks_num, block_dim)).astype(int)
    container = np.zeros(list(container_size)).astype(int)
    stable = [False] * blocks_num
    heightmap = np.zeros(container_size[:-1]).astype(int)
    valid_size = 0
    empty_size = 0

    # Initialize temporary variables for parallel tests
    positions_temp = np.zeros((blocks_num, block_dim)).astype(int)
    container_temp = np.zeros(list(container_size)).astype(int)
    stable_temp = [False] * blocks_num
    heightmap_temp = np.zeros(container_size[:-1]).astype(int)
    valid_size_temp = 0
    empty_size_temp = 0

    # Box matrix
    seen_blocks = blocks[blocks[:,0]<num_nodes,:]

    # Selection constraint mask
    selection_mask = np.ones(6*num_nodes).astype(bool)

    # While there are still boxes to place:
    for placement_index in range(blocks_num):

        if(not selection_mask.any()):
            print("All Boxes Packed")
            break

        # Initialize Score for Greedy picking
        best_C = 0.0
        best_block = 0

        # For every box in the Box matrix
        for block_index in range(len(seen_blocks)):

            # If it is still available:
            if(selection_mask[block_index]):

                # Fake temporary variables
                container_temp = container.copy()
                stable_temp = stable.copy()

                # Try to pack it in the current container just to see which score it would get (DO NOT UPDATE THE ACTUAL CONTAINER).
                # Here use the temporary variables which are "frozen" in the current environment and thus allow for all boxes to be
                # packed "in parallel" under the same scenario.
                _, _, _, heightmap_res, valid_size_res, _, placed = \
                    calc_one_position_lb_greedy_3d(seen_blocks[block_index,1:], placement_index, container_size,
                                                container_temp, positions_temp, stable_temp, heightmap_temp, valid_size_temp, empty_size_temp)

                # If the box could be packed in the current container given the contraints:
                if(placed==True):

                    # Calculate its WB-LB score
                    front_line = np.max(np.nonzero(heightmap_res)[0])+1
                    box_size_res = front_line * container_size[1] * container_size[2]
                    C = valid_size_res / box_size_res

                    # Greedily select the best block to pack form the Box matrix
                    if(C>best_C):
                        best_C = C
                        best_block = block_index

        # If no box could be packed:
        if(best_C == 0):
            break

        # Having Greedily chosen the best box to pack: now actually pack it and update the true variables
        container, positions, stable, heightmap, valid_size, empty_size, _ = \
                calc_one_position_lb_greedy_3d(seen_blocks[best_block,1:], placement_index, container_size,
                                            container, positions, stable, heightmap, valid_size, empty_size)

        # If in the last boxes:
        if(placement_index>=blocks_num-num_nodes):
            selected_box_and_rot = (np.arange(6)*num_nodes)+(best_block%num_nodes)
            selection_mask[selected_box_and_rot] = False
        # If not, rollout (update the box matrix with the next box form the full box sequence):
        else:
            seen_blocks[seen_blocks[:,0]==seen_blocks[best_block,0],:] = blocks[blocks[:,0]==(num_nodes+placement_index),:]

        # Update temporary variables to reflect the current container
        positions_temp = positions
        container_temp = container
        stable_temp = stable
        heightmap_temp = heightmap
        valid_size_temp = valid_size
        empty_size_temp = empty_size

    # Having finished packing the container:
    # Compute the Volume Utilization Reward from DBP (here called ratio)
    container_volume = container_size[0] * container_size[1] * container_size[2]
    ratio = np.exp(-(container_volume-valid_size) / container_volume)

    print("Ratio: ", ratio)

    stable_num = np.sum(stable)
    front_line = np.max(np.nonzero(heightmap)[0])+1
    box_size = front_line * container_size[1] * container_size[2]

    scores = [valid_size, box_size, empty_size, stable_num, np.max(heightmap)]

    lb_time = time.time() - lb_start
    # print("LB time: %2.4f" %lb_time)
    return positions, container, stable, ratio, scores


# ==================================================================================================== #
# =================================== AUXILIARY VISUALIZATION FUNCTIONS ============================== #
# ==================================================================================================== #

def calc_colors(max_len):
    mid_len = int(max_len/2)
    colors = np.zeros( (max_len, 3) )

    start_color = [237, 125, 49]
    # mid_color = [84, 130, 53]
    mid_color = [84, 250, 180]
    # end_color = [255, 217, 102]
    end_color = [160, 16, 40]
    
    for i in range(3): 
        colors[:mid_len, i] = np.linspace( start_color[i] , mid_color[i], mid_len)
        colors[mid_len:, i] = np.linspace( end_color[i] , end_color[i], max_len - mid_len)
    
    colors_str = []
    for color in colors:
        color = color.astype('int')
        colors_str.append( 
            '#%02X%02X%02X' % (color[0], color[1], color[2])
         )

    box_colors = ['#0b55b5', '#ffa600', '#4bab3c', '#e0e048', '#cc2f2f', '#50bdc9', 
                  '#a47c5f', '#321f91', '#edecf9', '#dd9286',  '#91aec6', 'silver',
                  'khaki', '#ada5de', 'coral', '#ffd64d', '#6e6702', 'yellowgreen',
                  'lightblue', 'salmon', '#004fa4', 'tan', '#fdff04', 'lavender', '#037272', 
                  '#eb1629', '#2a3132', '#336b87', '#f27657', '#598234', '#e53a3a', '#aebd38', '#c4dfe6',
                  '#44b8b1', '#ffbb00', '#fb6542', '#375e97', '#4cb5f5', '#f4cc70', '#1e434c', '#9b4f0f',
                  '#f1f1f2', '#bcbabe', '#20c0df', '#011a27', '#eed8ae', '#f0810f', '#4b7447', '#2d4262']

    return np.array(box_colors)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


colors = ['#ffd966', '#a9d08e', '#f4b084', '#9bc2e6', '#ff7c80', '#c6b5f0', 
    '#a0c8c0', '#f5f4c2', '#c0bed3', '#dd9286',  '#91aec6',
    'silver', 'khaki', 'lime', 'coral',
    'yellowgreen', 'lightblue', 'salmon', 'aqua', 'tan', 'violet', 'lavender',  ]

labels = np.array(labels)
colors = calc_colors(len(labels))

def get_cube_data(pos=[0,0,0], size=[1,1,1], face_index=[0]):
    l, w, h = size
    a, b, c = pos
    x = [ [a, a + l, a + l, a, a],
            [a, a + l, a + l, a, a],
            [a, a + l, a + l, a, a],
            [a, a + l, a + l, a, a]
        ]
    y = [ [b, b, b + w, b + w, b],
            [b, b, b + w, b + w, b],  
            [b, b, b, b, b],
            [b + w, b + w, b + w, b + w, b + w]
             ]   
    z = [ [c, c, c, c, c],                       
            [c + h, c + h, c + h, c + h, c + h],
            [c, c, c + h, c + h, c],
            [c, c, c + h, c + h, c] ]
    x = np.array(x)[face_index]
    y = np.array(y)[face_index]
    z = np.array(z)[face_index]
    return x, y, z

def draw_container_voxel(container, blocks_num, colors=colors, 
    order=None, 
    rotate_state=None,
    feasibility=None, 
    draw_top = True,
    view_type='front',
    blocks_num_to_draw=None,
    save_name='result'):

    colors = calc_colors(blocks_num)

    if blocks_num_to_draw is None:
        blocks_num_to_draw = blocks_num
    container_width = container.shape[0]
    container_length = container.shape[1]
    container_height = container.shape[2]

    if rotate_state is None:
        rotate_state = np.zeros(blocks_num)

    if order is None:
        order = [i for i in range(blocks_num)]
        
    rotate_state = np.array(rotate_state)

    edges_not_rotate_color = np.empty( np.sum(rotate_state == False)  ).astype('object')
    edges_rotate_color = np.empty( np.sum(rotate_state == True) ).astype('object')

    for i in range(len(edges_not_rotate_color)):
        # edges_not_rotate_color[i] = '#00225515'
        edges_not_rotate_color[i] = '#00225500'
    for i in range(len(edges_rotate_color)):
        # edges_rotate_color[i] = '#002255'
        edges_rotate_color[i] = '#00225500'

    blocks_color = np.empty_like(container).astype('object')
    voxels = np.zeros_like(container).astype('bool')

    voxels_rotate = np.zeros_like(container).astype('bool')
    voxels_not_rotate = np.zeros_like(container).astype('bool')

    place_order = []
    for i in range(blocks_num_to_draw):
        block_index = order[i]
        block_voxel = (container == i+1)

        blocks_color[block_voxel] = colors[block_index%49]

        # if rotate_state is not None:
        if rotate_state[i] == True:
            voxels_rotate = voxels_rotate | block_voxel
        else:
            voxels_not_rotate = voxels_not_rotate | block_voxel

        place_order.append(block_index)


    plt.close('all')
    fig = plt.figure( figsize=(3,5) )
    fig.subplots_adjust(left=0, right=1, bottom=-0.00)
    
    ax = Axes3D(fig)

    xticks = [ i for i in range(0, container_width+1)]
    yticks = [ i for i in range(0, container_length+1)]
    zticks = [ i for i in range(0, container_height+1)]
    xlabels = [ '' for i in range(0, container_width+1)]
    ylabels = [ '' for i in range(0, container_length+1)]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    
    ax.set_xlim3d(0, container_width)
    ax.set_ylim3d(0, container_length)
    ax.set_zlim3d(0, container_height)

    plt.grid(True, alpha=0.3, lw=1 )

    ax.set_axisbelow(True)

    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    zorder = 2000
    w, l, h = container.shape

    ax.plot([w,w], [0,0], [0,h], 'k-', linewidth=1, zorder=zorder)
    ax.plot([0,0], [0,0], [0,h], 'k-', linewidth=1, zorder=-1)
    if view_type == 'front':
        ax.plot([w,w], [l,l], [0,h], 'k-', linewidth=1, zorder=-1)
        ax.plot([0,0], [l,l], [0,h], 'k-', linewidth=1, zorder=zorder)
    else:
        # ax.plot([w,w], [l,l], [0,h], 'b-', linewidth=1, zorder=zorder)
        ax.plot([0,0], [l,l], [0,h], 'k-', linewidth=1, zorder=-1)
    # top
    top_h = h
    for z in range(h):
        if (container[:,:,z] == 0).all():
            top_h = z
            break

    gap = 0.0
    X, Y, Z = get_cube_data(pos=[-gap,-gap,top_h], 
        size=[container_width + 2*gap, container_length + 2*gap, 0.001], 
        face_index=[0, 2])
    
    # bottom
    ax.plot([0,w], [0,0], [0,0], 'k-', linewidth=1, zorder=-1)
    ax.plot([0,w], [l,l], [0,0], 'k-', linewidth=1, zorder=-1)
    if view_type == 'front':
        ax.plot([0,0], [0,l], [0,0], 'k-', linewidth=1, zorder=zorder)
        ax.plot([w,w], [0,l], [0,0], 'k-', linewidth=1, zorder=-1)
    else:
        ax.plot([0,0], [0,l], [0,0], 'k-', linewidth=1, zorder=-1)
        ax.plot([w,w], [0,l], [0,0], 'k-', linewidth=1, zorder=zorder)
    
    if draw_top:
        zlabels = [ str(i) if i==top_h else '' for i in range(0, container_height+1)]
    else:
        zlabels = [ '' for i in range(0, container_height+1)]

    ax.voxels(voxels_rotate, facecolors=blocks_color, edgecolor=edges_rotate_color, alpha=1, zorder=-1, shade=False)
    ax.voxels(voxels_not_rotate, facecolors=blocks_color, edgecolor=edges_rotate_color, alpha=1, zorder=-1, shade=False)

    if view_type == 'front':
        ax.view_init(13, -130) #(13, -130)
        plt.savefig(save_name + '-' + view_type + '.png', bbox_inches=0, dpi=400)
    else:
        ax.view_init(26, 40)
        plt.savefig(save_name + '-' + view_type + '.png', bbox_inches=0, dpi=400)
        
