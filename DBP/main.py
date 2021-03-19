"""

Deep Box Packing - Main: Main file for Training & Validating DBP

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
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.signal
import random

from DBP import DBP
import draw

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ==================================================================================================== #
# ======================================== AUXILIARY FUNCTIONS ======================================= #
# ==================================================================================================== #

# This function is only for the argparser to convert strings into boolean arguments
def str2bool(v):
      return v.lower() in ('true', '1')

# This function was imported from the PPO core.py file form OpenAI Spinningup repository available at 
# https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/ppo.
def discount_sum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    z = x.detach().numpy()
    disc_sum = np.zeros_like(z)
    for index in range(z.shape[0]):
        disc_sum[index] = scipy.signal.lfilter([1], [1, float(-discount)], z[index,::-1], axis=0)[::-1]

    return torch.from_numpy(disc_sum)

# ==================================================================================================== #
# ============================= MASK DICTIONARIES INITIALIZATION FUNCTIONS =========================== #
# ==================================================================================================== #

# This function builds Mask Dictionaries for the Border, Height and Selection Masks. Nothe that these
# are NOT the Border, Height and Selection masks. These Dictionaries serve as ways to speed up 
# computations in training/ test time. Instead of fully computing these masks at run time, all 
# computations that can be can be done offline are already prepared here in these Dictionaries.
# Therefore, when it comes the time to update the masks during the packing procedure, the calculations are
# reduced to lookup processes in these dictionaries. See that these dictionaries are constant throughout
# a given packing, while only their respective masks are constantly updated.

def build_mask_dicts(num_boxes, container_width, container_length, max_size):

    #Max bottom area of a box
    box_shapes = ((max_size-1)*(max_size-1))

    #Bottom area of the container
    map_size = container_width*container_length

    #Size of the masks (6*n*W*L), same as the action probability vector
    mask_size = num_boxes*6*container_width*container_length

    keys_map = np.arange(max_size)+1
    keys_select = np.arange(num_boxes)
    keys_height = np.arange(box_shapes*map_size)

    map_idx = np.arange(map_size).reshape(container_width, container_length)
    select_idx = np.arange(map_size*6)*num_boxes

    rows_idx = map_idx[container_width-max_size+1:, :].flatten()
    cols_idx = (np.transpose(map_idx))[container_length-max_size+1:, :].flatten()

    dict_border_rows = dict.fromkeys(keys_map, []) 
    dict_border_cols = dict.fromkeys(keys_map, []) 
    dict_select = dict.fromkeys(keys_select, [])   
    dict_height = dict.fromkeys(keys_height, [])

    # Build border dictionaries
    #
    # rows - keys: each possible box width
    #        values: which rows of heightmap are not viable placements
    # cols - keys: each possible box length
    #        values: which cols of heightmap are not viable placements
    for item in range(2,max_size+1):
        dict_border_rows[item] = rows_idx[container_length*(max_size-item):]
        dict_border_cols[item] = cols_idx[container_width*(max_size-item):]

    # Build selection dictionary
    #
    # keys: each box in the Information window
    # values: every index in the 6*n*W*L vector associated with it [[box in 1st orientation,...,box in 6th orientation] in 1st position,...]
    for item in range(0,num_boxes):
        dict_select[item] = select_idx+item

    # Build height dictionary
    #
    # keys: each possible position-box_bottom_area combination
    # values: which indeces would box w*l not be able to be placed on if the position specified by the key would be saturated
    # (a saturated cell is a cell whose current value, summed with a box' height, would exceed the maximum height of the container)
    for item in range(keys_height.shape[0]):

        dict_height[item] = np.ones(mask_size)

        index_w = int((item/box_shapes)/container_length)
        index_l = int((item/box_shapes)%container_length)

        w = int((item%box_shapes)/(max_size-1)+1)
        l = int((item%box_shapes)%(max_size-1)+1)

        w_a = np.arange(np.clip(index_w-w+1,0,None), index_w+1)
        w_b = np.clip(l,None,index_l+1)
        w_hm = w_a.repeat(w_b)

        l_a = np.arange(np.clip(index_l-l+1,0,None), index_l+1)
        l_b = np.clip(w,None,index_w+1)
        l_hm = np.tile(l_a,l_b)

        indices = np.zeros((2,w_hm.shape[0]))

        indices[0]=w_hm
        indices[1]=l_hm

        dict_height[item] = indices

    mask_dicts = [dict_border_rows, dict_border_cols, dict_select, dict_height]

    return mask_dicts

# This functions builds the Stability Dictionary which holds, for each possible type of box 
# bottom area, the stable positions in which it can be packed in the container. Note that this
# is the only Mask Dictionary which is updated throughout the packing process along with its mask.
def build_stab_dict(batch_size, c_width, c_length, max_size):

    # Bottom area of the container
    map_size = c_width*c_length

    # Each key of the Dictionary is a number indexing a possible box bottom area
    keys_stab = np.arange((max_size-1)*(max_size-1))
    stab_dict = dict.fromkeys(keys_stab, [])

    # Initially, all positions are stable for any type of box (hence, fill every position with ones)
    for k,v in stab_dict.items():
        stab_dict[k] = np.ones((batch_size,map_size))

    return stab_dict

# This functions resets the Stability Dictionary to all ones every time packings are done
def reset_stab_dict(stab_dict):

    for k,v in stab_dict.items():
        stab_dict[k] = np.ones_like(stab_dict[k])

    return stab_dict

# ==================================================================================================== #
# ======================================== VALIDATION FUNCTION ======================================= #
# ==================================================================================================== #

# This is the Validation function called at the end of every training round for evaluating results
def validate(indices, actor, valid_dir, epoch, mask_dicts, **kwargs):
    """
    input: 
        indices: which validations samples to use
        actor: Model used for computing actions (DBP)
        valid_dir: Directory to save validation results
        epoch: which epoch the system is validating
        mask_dicts: Masks dictionaries to be passed to the model (DBP)
        kwargs: global arguments

    output:
        Mean of total validation reward, Mean of Volume Utilization reward
        
    """

    #Set Network in evaluation mode
    actor.eval()

    #Initialize the Stability dictionary
    stab_dict = build_stab_dict(kwargs['valid_size'], kwargs['container_width'], kwargs['container_length'], kwargs['max_size'])

    # Volume Utilization Rewards
    rewards_V = []
    # Total Rewards (Volume Utilization and Constructive, averaged)
    rewards_mean = []

    for batch_idx in range(1):

        # Input data (note that |full_box_sequence| = 6 * total number of boxes in the conveyor belt)
        full_box_sequence, heightmap = kwargs['valid_data'].__getitem__(indices)

        # Full forward pass through the dataset
        with torch.no_grad():
            # Pack full_box_sequence into heightmap
            logp, reward, value, V_rew, draw_info = actor(full_box_sequence, heightmap, mask_dicts, stab_dict, kwargs['access'], draw=True)
            # Reset the Stability Dictionary
            stab_dict = reset_stab_dict(stab_dict)

        # Average and store results
        reward_V = V_rew.mean().item()
        std_V = V_rew.std().item()
        rewards_V.append(reward_V)

        reward_mean = reward.mean().item()
        rewards_mean.append(reward_mean)

        # Visualize results
        if kwargs['render_fn'] is not None:

            save_dir = os.path.join(valid_dir, '%s__%2.4f__%2.4f' % (epoch,reward_V,std_V))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            name = 'batch%d'%batch_idx
            path = os.path.join(save_dir, name)
            kwargs['render_fn'](draw_info, V_rew, path, **kwargs)

    # Set Network back into training mode
    actor.train()

    return np.mean(rewards_mean), np.mean(rewards_V)


# ==================================================================================================== #
# ========================================= TRAINING FUNCTION ======================================== #
# ==================================================================================================== #

def train(actor, **kwargs):

    # Time measure settings
    date = datetime.datetime.now()
    now = '%s' % date.date()
    now += '-%s' % date.hour
    now += '-%s' % date.minute
    now = str(now)
    print("Training initiated at: ", now)
    print("")

    # Save Directories settings
    save_dir = os.path.join('results', '%d' % kwargs['iw'], 
        str(kwargs['title']) + "-Stable_" + str(kwargs['stable_placements']) + '-' + str(kwargs['activity']) + '-' + str(kwargs['total_obj_num']) + '-' + str(kwargs['train_size']) + '-' + str(kwargs['batch_size']) + '-' + str(kwargs['embed_size']) + '-' + str(kwargs['num_layers']) + '-' + str(kwargs['forward_expansion']) + '-' + str(kwargs['heads']) + '-' + kwargs['rew_format'] + '-' + str(kwargs['actor_lr']) + '-' + now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    valid_dir = os.path.join(save_dir, 'render')#, '%s' % epoch)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    # Optimizer
    actor_optim = optim.Adam(actor.parameters(), lr=kwargs['actor_lr'])

    # Initialize full sequence indeces and global buffers
    train_indices = np.arange(kwargs['train_size'])
    valid_indices = np.arange(kwargs['valid_size'])

    best_reward = np.inf

    my_rewards = []
    my_V_rewards = []

    my_act_losses = []
    my_crit_losses = []

    train_size = kwargs['train_size']

    # Step size for logging info
    log_step = int(train_size / kwargs['batch_size'])
    if log_step > 100:
        log_step = int(100)
    if log_step == 0:
        log_step = int(1)

    log_step=10

    #Build Mask Dictionaries
    mask_dicts = build_mask_dicts(kwargs['iw'], kwargs['container_width'], kwargs['container_length'], kwargs['max_size'])
    stab_dict = build_stab_dict(kwargs['batch_size'], kwargs['container_width'], kwargs['container_length'], kwargs['max_size'])

    # ==================================================== Training Loop ==================================================== #

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(kwargs['epoch_num']):

        epoch_start = time.time()
        start = epoch_start

        if(kwargs['activity']=='training'):

            #Set Network in training mode
            actor.train()

            # Initialize local buffers
            times, act_losses, rewards, rewards_V, crit_losses = [], [], [], [], []

            # Prepare batch
            np.random.shuffle(train_indices)

            # For every batch in the data, do:
            for batch_idx in range(int(kwargs['train_size']/kwargs['batch_size'])):

                # Get input data (note that |full_box_sequence| = 6 * total number of boxes in the conveyor belt)
                full_box_sequence, heightmap = kwargs['train_data'].__getitem__(train_indices[batch_idx*kwargs['batch_size']:(batch_idx+1)*kwargs['batch_size']])

                use_cuda = kwargs['use_cuda']
                if use_cuda:
                    full_box_sequence = full_box_sequence.cuda().detach()
                    heightmap = heightmap.cuda().detach()

                # Pack full_box_sequence into heightmap
                logp, reward, value, V_rew, draw_info = actor(full_box_sequence, heightmap, mask_dicts, stab_dict, kwargs['access'], draw=False)
                # Reset the Stability Dictionary
                stab_dict = reset_stab_dict(stab_dict)

                value = value.squeeze()

                # ============================= VPG Loss Functions ============================= #

                # GAE-Lambda Advantage Estimation
                gamma = 0.998
                lamb = 0.95

                deltas = reward[:,:-1] + gamma * value[:,1:] - value[:,:-1]
                last_delta = (reward[:,-1]-value[:,-1]).reshape(deltas.shape[0],1)
                deltas = torch.cat((deltas, last_delta),1)
                advantage = discount_sum(deltas, gamma * lamb)

                # Rewards-to-go computation
                rewards_to_go = discount_sum(reward, gamma)

                # Critic Loss
                v_regress = (rewards_to_go - value)
                critic_loss = torch.mean(v_regress ** 2)

                # Actor Loss
                a_regress = (advantage.detach() * logp).flatten()
                # only backpropagate on valid actions (this is necessary because if a container is 
                # fully packed but others are still being packed, void actions are added to it for a
                # matter of data structure consistency)
                a_regress_select = a_regress.clone()[~torch.isinf(a_regress)]
                actor_loss = torch.mean(a_regress_select)
                
                full_loss = actor_loss + critic_loss

                # ================================ Optimization =============================== #

                actor_optim.zero_grad()
                full_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), kwargs['max_grad_norm'])
                actor_optim.step()

                # ================================ Final Logging =============================== #

                rewards_V.append(torch.mean(V_rew.detach()).item())
                rewards.append(torch.mean(reward.detach()).item())
                act_losses.append(torch.mean(actor_loss.detach()).item())
                crit_losses.append(torch.mean(critic_loss.detach()).item())

                print("Batch %d: done" %batch_idx)

                if (batch_idx + 1) % log_step == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end

                    mean_crit_loss = np.mean(crit_losses[-log_step:])
                    mean_act_loss = np.mean(act_losses[-log_step:])
                    mean_reward = np.mean(rewards[-log_step:])
                    mean_rewards_V = np.mean(rewards_V[-log_step:])
                    my_rewards.append(mean_reward)
                    my_V_rewards.append(mean_rewards_V)
                    my_act_losses.append(mean_act_loss)
                    my_crit_losses.append(mean_crit_loss)

                    print('Epoch %d  Batch %d/%d, mean_reward: %2.3f, mean reward_V: %2.3f, crit loss: %2.4f, took: %2.4fs' %
                          (epoch, batch_idx, kwargs['train_size']/kwargs['batch_size'], mean_reward, mean_rewards_V, mean_crit_loss, times[-1]))


            mean_act_loss = np.mean(act_losses)
            mean_crit_loss = np.mean(crit_losses)
            mean_reward = np.mean(rewards)

            # Save the Network weights
            epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)

            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

        # ======================================================= Validation ======================================================= #

        mean_valid_rew, mean_valid_V = validate(
            valid_indices, 
            actor,
            valid_dir,
            epoch,
            mask_dicts,
            **kwargs
        )

        # Save best model parameters
        if mean_valid_rew < best_reward:

            best_reward = mean_valid_rew

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

        print('    mean valid reward: %2.4f, mean valid V reward: %2.4f, took: %2.4fs ' % (mean_valid_rew, mean_valid_V, time.time() - epoch_start))
        print("\n")

        # ================================================== Update training plots ================================================== #

        import matplotlib.pyplot as plt
        plt.close('all')
        plt.title('Reward')
        plt.plot(range(len(my_rewards)), my_rewards, '-')
        plt.savefig(save_dir + '/reward.png' , bbox_inches='tight', dpi=400)

        plt.close('all')
        plt.title('V Reward')
        plt.plot(range(len(my_V_rewards)), my_V_rewards, '-')
        plt.savefig(save_dir + '/V_reward.png' , bbox_inches='tight', dpi=400)

        plt.close('all')
        plt.title('Critic Loss')
        plt.plot(range(len(my_crit_losses)), my_crit_losses, '-')
        plt.savefig(save_dir + '/critic_losses.png', bbox_inches='tight', dpi=400)

        if(kwargs['activity']!='training'): 
            break

    # Save results on text files for posterior evaluation
    np.savetxt(save_dir + '/rewards.txt', my_rewards)
    np.savetxt(save_dir + '/V_rewards.txt', my_V_rewards)
    np.savetxt(save_dir + '/act_losses.txt', my_act_losses)
    np.savetxt(save_dir + '/crit_losses.txt', my_crit_losses)
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.title('Reward')

    plt.plot(range(len(my_rewards)), my_rewards, '-')
    plt.savefig(save_dir + '/reward.png', bbox_inches='tight', dpi=400)

    plt.close('all')
    plt.title('Critic Loss')
    plt.plot(range(len(my_crit_losses)), my_crit_losses, '-')
    plt.savefig(save_dir + '/critic_losses.png', bbox_inches='tight', dpi=400)

# ==================================================================================================== #
# ========================================= MAIN PACK FUNCTION ======================================= #
# ==================================================================================================== #

def pack(**kwargs):

    import data
    from data import PACKDataset

    print('Loading data...')
    use_cuda = kwargs['use_cuda']

    size_range = [ kwargs['min_size'], kwargs['max_size'] ]

    # Create Training and Validation files
    train_file, valid_file = data.create_dataset(
        kwargs['total_obj_num'],
        kwargs['train_size'],
        kwargs['valid_size'],
        kwargs['obj_dim'],
        size_range,
        seed=kwargs['seed'],
    )

    print(train_file)
    print(valid_file)

    # Create Training and Validation Datasets (full box sequences & heightmaps)
    train_data = PACKDataset(
        train_file,
        kwargs['total_obj_num'],
        kwargs['train_size'],
        kwargs['seed'],
        kwargs['heightmap_type'],
        kwargs['container_width'],
        kwargs['container_length'],
        kwargs['obj_dim'],
        unit=kwargs['unit'],
        )

    valid_data = PACKDataset(
        valid_file,
        kwargs['total_obj_num'],
        kwargs['valid_size'],
        kwargs['seed'] + 1,
        kwargs['heightmap_type'],
        kwargs['container_width'],
        kwargs['container_length'],
        kwargs['obj_dim'],
        unit=kwargs['unit'],
        )

    # DBP setup
    network = DBP

    actor = network(kwargs['iw'],
                kwargs['embed_size'],
                kwargs['forward_expansion'],
                kwargs['heads'],
                kwargs['use_cuda'],
                kwargs['container_width'],
                kwargs['container_length'],
                kwargs['container_height'],
                kwargs['heightmap_type'],
                kwargs['max_size'],
                kwargs['num_layers'],
                kwargs['dropout'],
                kwargs['unit'],
                kwargs['stable_placements']
                )

    if use_cuda:
        actor = actor.cuda()

    # Final settings
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['render_fn'] = draw.draw_container

    if kwargs['checkpoint']:
        path = os.path.join(kwargs['checkpoint'], 'actor.pt')
        actor.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

        print('Loading pre-train model', path)

    # Train
    train(actor, **kwargs)


# ============================================================================================================================================ #
# ================================================================== MAIN CALL =============================================================== #
# ============================================================================================================================================ #

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DBP')

    # Task settings
    parser.add_argument('--title', default='DBP')                                   # Title of results folder
    parser.add_argument('--activity', default='training', type=str)                 # training / testing
    parser.add_argument('--use_cuda', default=True, type=str2bool)                  # Parameter for cuda enabled hardware (not used)
    parser.add_argument('--cuda', default='0', type=str)                            # Cuda parameter (not used)
    parser.add_argument('--cpu_threads', default=0, type=int)                       # Number of CPU threads
    parser.add_argument('--checkpoint', default=None)                               # Optional past experiment form which to load network weights
    parser.add_argument('--seed', default=12345, type=int)                          # Random seed

    # Training/testing settings
    parser.add_argument('--train_size',default=1280, type=int)                      # Number of Full box sequences in the training set
    parser.add_argument('--valid_size', default=10, type=int)                       # Number of Full box sequences in the validation set
    parser.add_argument('--epoch_num', default=2, type=int)                         # Number of epochs for which to train DBP
    parser.add_argument('--batch_size', default=128, type=int)                      # Size of batches

    # Data settings
    parser.add_argument('--obj_dim', default=3, type=int)                           # Three-dimensional boxes
    parser.add_argument('--information_window', dest='iw', default=10, type=int)    # Size of the Information Window
    parser.add_argument('--total_obj_num', default=150, type=int)                   # Number of boxes in a Full box sequence

    # sizes of blocks and containers
    parser.add_argument('--unit', default=1.0, type=float)                          # Size of a discretized unit
    parser.add_argument('--min_size', default=1, type=int)                          # Minimum size of box side
    parser.add_argument('--max_size', default=4, type=int)                          # (Maximum size + 1) of box side
    parser.add_argument('--container_width', default=8, type=int)                   # Container width (from the open entrance to the back)
    parser.add_argument('--container_length', default=6, type=int)                  # Container length (from left side wall to right side wall)
    parser.add_argument('--container_height', default=6, type=int)                  # Container Height

    # Packing settings
    parser.add_argument('--heightmap_type', default='full', type=str)               # full = every cell holds the absolute height value (others at TAP-Net)
    parser.add_argument('--rew_format', default='Constructive-Utilization')         # Name of the reward function used (dummy, only serves for logging)
    parser.add_argument('--stable_placements', default=True, type=str2bool)        # Whether boxes can only to packed with full support
    parser.add_argument('--access', default=False, type=str2bool)                   # Whether boxes can only be placed in accessible positions

    # Network parameters    
    parser.add_argument('--dropout', default=0.1, type=float)                       # Dropout parameter for the Mapping Transformer network (MpT)
    parser.add_argument('--actor_lr', default=5e-4, type=float)                     # Learning rate for the optimizer
    parser.add_argument('--max_grad_norm', default=2., type=float)                  # Gradient clipping parameter
    parser.add_argument('--num_layers', dest='num_layers', default=6, type=int)     # Number of Encoder/ Decoder Layers in the MpT
    parser.add_argument('--embed_size', dest='embed_size', default=32, type=int)    # Size of boxes and positions embedding
    parser.add_argument('--forward_expansion', default=4, type=int)                 # Ratio by which to temporarily enlarge the embedding space in the Feed Forward layer of the MpT
    parser.add_argument('--heads', default=4, type=int)                             # Number of Attention heads in the Multi-Head attention blocks of the MpT

    args = parser.parse_args()

    if args.cpu_threads != 0:
        torch.set_num_threads(args.cpu_threads)

    print('Reward type: %s' % args.rew_format)
    print('Full box sequence size: %s'  % args.total_obj_num)
    print('Information window size: %s'  % args.iw)
    print('Container width: %s'  % args.container_width)
    print('Container length: %s' % args.container_length)
    print('Container height: %s' % args.container_height)
    print('activity: %s' % args.activity)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    kwargs = vars(args)
    pack(**kwargs)

    