"""

Wall Building LB-Greedy - Main: Main file for callind WB-LB-Greedy

Author: Leonardo Albuquerque - ETH Zürich, 2021
Based on TAP-Net's original code from https://vcc.tech/research/2020/TAP

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import os
import time
import argparse
import datetime
import numpy as np

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ==================================================================================================== #
# ======================================== AUXILIARY FUNCTIONS ======================================= #
# ==================================================================================================== #

# This function is only for the argparser to convert strings into boolean arguments
def str2bool(v):
      return v.lower() in ('true', '1')

# This is the call to the Wall-Building Heuristic
def heuristic(**kwargs):

    ## Time measure settings
    date = datetime.datetime.now()
    now = '%s' % date.date()
    now += '-%s' % date.hour
    now += '-%s' % date.minute
    now = str(now)

    ## Save Directories settings
    save_dir = os.path.join('results', '%d' % kwargs['iw'], 
        str(kwargs['obj_dim']) + 'd-'  + kwargs['reward_type'] + '-' + kwargs['packing_strategy'] + '-width-' + str(kwargs['container_width']) + '-' + now)

    ## Batch loader
    loader = DataLoader(kwargs['data'], len(kwargs['data']), shuffle=False, num_workers=0)

    # Initialize local buffers
    times, losses, rewards, critic_rewards = [], [], [], []

    epoch_start = time.time()

    # Render Directories settings
    heuristic_dir = os.path.join(save_dir, 'render')
    if not os.path.exists(heuristic_dir):
        os.makedirs(heuristic_dir)

    rewards = []
    stds = []

    # For every batch in the data, do:
    for batch_idx, batch in enumerate(loader):

        # Get input data
        full_box_sequence, heightmap = batch

        name = 'batch%d.png'%(batch_idx)
        path = os.path.join(heuristic_dir, name)

        # Pack full_box_sequence
        mean_reward, std_reward = kwargs['render_fn'](full_box_sequence, path, **kwargs)

        # Log results
        rewards.append(mean_reward)
        stds.append(std_reward)

    # Average results
    mean_rew = np.mean(rewards)
    std_rew = np.mean(stds)

    print('Mean reward: %2.4f,  Std reward: %2.4f, took: %2.4fs '%(mean_rew, std_rew, time.time() - epoch_start))

def run(**kwargs):

    import pack
    import data
    from data import PACKDataset

    size_range = [ kwargs['min_size'], kwargs['max_size'] ]

    # Create Training and Validation files
    data_file = data.create_dataset(
        kwargs['total_obj_num'],
        kwargs['seq_number'],
        kwargs['obj_dim'],
        size_range,
        seed=kwargs['seed'],
    )

    print(data_file)

    # Create Dataset
    data = PACKDataset(
        data_file,
        kwargs['total_obj_num'],
        kwargs['seq_number'],
        kwargs['seed'] + 1,
        kwargs['heightmap_type'],
        kwargs['container_width'],
        kwargs['container_length'],
        kwargs['obj_dim'],
        unit=kwargs['unit'],
        )

    kwargs['data'] = data
    kwargs['render_fn'] = pack.pack_and_render

    # Pack
    heuristic(**kwargs)

# ============================================================================================================================================ #
# ================================================================== MAIN CALL =============================================================== #
# ============================================================================================================================================ #

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='WB-LB-Greedy')

    # Task settings
    parser.add_argument('--seed', default=12345, type=int)                          # Random seed

    # Training/testing settings
    parser.add_argument('--seq_number',default=100, type=int)                       # Number of Full box sequences to be packed

    # Data settings
    parser.add_argument('--obj_dim', default=3, type=int)                           # Three-dimensional boxes
    parser.add_argument('--information_window', dest='iw', default=10, type=int)    # Size of the Information Window
    parser.add_argument('--total_obj_num', default=10, type=int)                    # Number of boxes in a Full box sequence                   

    # sizes of blocks and containers
    parser.add_argument('--unit', default=1.0, type=float)                          # Size of a discretized unit
    parser.add_argument('--min_size', default=1, type=int)                          # Minimum size of box side
    parser.add_argument('--max_size', default=4, type=int)                          # (Maximum size + 1) of box side
    parser.add_argument('--container_width', default=8, type=int)                   # Container width (from the open entrance to the back)
    parser.add_argument('--container_length', default=6, type=int)                  # Container length (from left side wall to right side wall)
    parser.add_argument('--container_height', default=6, type=int)                  # Container Height

    # Packing settings
    parser.add_argument('--packing_strategy', default='LB_GREEDY', type=str)        # Algorithm according to which to pack boxes
    parser.add_argument('--reward_type', default='comp', type=str)                  # Type of Reward function used
    parser.add_argument('--heightmap_type', default='full', type=str)               # Type of Container representation

    args = parser.parse_args()

    kwargs = vars(args)
    run(**kwargs)
