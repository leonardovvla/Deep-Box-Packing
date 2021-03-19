"""

Deep Box Packing - Data: File for generating and structuring the input data for DBP

Author: Leonardo Albuquerque - ETH Zürich, 2021
Based on TAP-Net's original code from https://vcc.tech/research/2020/TAP

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import itertools

# ==================================================================================================== #
# =========================================== DATASET CLASS ========================================== #
# ==================================================================================================== #

# Class which creates the Box Sequences and the Heightmaps
class PACKDataset(Dataset):
    def __init__(self, data_file, blocks_num, num_samples, seed, heightmap_type, container_width, container_length, obj_dim, unit=1):

        super(PACKDataset, self).__init__()
        if seed is None:
            seed = np.random.randint(123456)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ============================ Build Full box sequence ============================ #

        blocks = np.loadtxt(data_file + 'blocks.txt').astype('float32')

        block_dim = obj_dim
        rotate_types = np.math.factorial(block_dim)

        blocks = blocks.reshape( num_samples, -1, block_dim, blocks_num)
        # num_samples x rotate_types x blocks_num x block_dim
        blocks = blocks.transpose(0, 1, 3, 2)
        # num_samples x (rotate_types * blocks_num) x block_dim
        blocks = blocks.reshape( num_samples, -1, block_dim )
        # num_samples x block_dim x (blocks_num * rotate_types)
        blocks = blocks.transpose(0,2,1)
        blocks = torch.from_numpy(blocks)

        # resolution
        blocks = blocks * unit
        # if unit<1:
        blocks = blocks.ceil()#.int()

        blocks_index = torch.arange(blocks_num)
        blocks_index = blocks_index.unsqueeze(0).unsqueeze(0)
        # num_samples x 1 x (blocks_num * rotate_types)
        blocks_index = blocks_index.repeat(num_samples, 1, rotate_types).float()

        # num_samples x (1 + block_dim) x (blocks_num * rotate_types)
        self.full_box_matrix = torch.cat( (blocks_index, blocks), 1 )

        # ============================ Build Heightmap ============================ #

        heightmap_num = 1
        
        if heightmap_type == 'diff':
            heightmap_num = 2
            heightmap_width = container_width * unit
            heightmap_length = container_length * unit
        else:
            heightmap_width = container_width * unit
            heightmap_length = container_length * unit

        # if unit < 1:
        heightmap_width = np.ceil(heightmap_width).astype(int)
        heightmap_length = np.ceil(heightmap_length).astype(int)

        self.heightmap = torch.zeros(num_samples, heightmap_num, heightmap_width, heightmap_length, requires_grad=True)

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.full_box_matrix[idx], self.heightmap[idx])

# ==================================================================================================== #
# ==================================== DATA GENERATION FUNCTIONS ===================================== #
# ==================================================================================================== #

# Auxiliary function for generating boxes from a probability distribution of possible sizes
def generate_blocks(blocks_num, size_range, obj_dim=3):

    block_dim = obj_dim

    min_box = size_range[0]
    max_box = size_range[1]
    size_list = [ i for i in range(min_box, max_box) ]

    # Distribution form which to sample to sizes of the sides of the boxes
    if len(size_list) == 3:
        prob_blocks = [0.45, 0.45, 0.10]
    elif len(size_list) == 4:
        prob_blocks = [0.25, 0.35, 0.35, 0.05]
    elif len(size_list) == 5:
        prob_blocks = [0.08, 0.26, 0.32, 0.26, 0.08]
    else:
        mu = 0.5
        sigma = 0.16
        prob_x = np.linspace(mu - 3*sigma, mu + 3*sigma, len(size_list))
        prob_blocks = np.exp( - (prob_x-mu)**2 / (2*sigma**2) ) / (np.sqrt(2*np.pi) * sigma)
        prob_blocks = prob_blocks / np.sum(prob_blocks)
    
    blocks = np.random.choice(size_list, (blocks_num, block_dim), p=prob_blocks )

    rotate_blocks = []
    blocks = blocks.transpose()

    for p in itertools.permutations( range(block_dim) ):
        rotate_blocks.append( blocks[list(p)].flatten() )

    rotate_blocks = np.array(rotate_blocks)

    return rotate_blocks

# Function for creating the training and validation box sets
def create_dataset( blocks_num, seq_number, obj_dim, size_range, seed=None):
    blocks_num = int(blocks_num)
    if seed is None:
        seed = np.random.randint(123456789)
    np.random.seed(seed)

    valid_dir = './data/rand_3d/pack-valid-' + str(blocks_num) + '-' + str(seq_number) + '-' + str(size_range[0]) + '-' + str(size_range[1]) + '/'

    def arr2str(arr):
        ret = ''
        for i in range(len(arr)-1):
            ret += str(arr[i]) + ' '
        ret += str(arr[-1]) + '\n'
        return ret

    if os.path.exists(valid_dir  + 'blocks.txt'):
        return valid_dir

    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
        
    def generate_data(data_dir, obj_dim, data_size):

        blocks_f = open(data_dir + 'blocks.txt', 'w')

        for _ in tqdm(range(data_size)):

            # continue
            rotate_blocks = \
                generate_blocks(blocks_num, size_range, obj_dim)

            for blocks_index, blocks in enumerate(rotate_blocks):
                blocks_f.writelines(arr2str( blocks ) )

        blocks_f.close()

    if not os.path.exists(valid_dir + 'blocks.txt'):
        generate_data(valid_dir, obj_dim, seq_number)
    return valid_dir
