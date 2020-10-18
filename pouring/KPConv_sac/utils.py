import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
from collections import deque
import pouring.KPConv_sac.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import pouring.KPConv_sac.cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

from pouring.KPConv_sac.kernel_points import create_3D_rotations


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


class ReplayBufferPointCloud():
    """Buffer to store environment transitions."""

    def __init__(self, capacity, batch_size, config, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.config = config 

        self.obses = [] # point cloud, use a list to store
        self.next_obses = []
        self.actions = np.empty((capacity, config.action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.reduced_states = np.empty((capacity, config.reduced_state_dim), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        reduced_state = None
        if isinstance(obs, tuple):
            reduced_state = obs[1]
            obs = obs[0]
            next_obs = next_obs[0]

        self.obses.append(obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        self.next_obses.append(next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        if len(self.obses) > self.capacity:
            self.obses.pop(0)
        if len(self.next_obses) > self.capacity:
            self.next_obses.pop(0)

        if reduced_state is not None:
            np.copyto(self.reduced_states[self.idx], reduced_state)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = []
        next_obses = []
        obs_features = []
        next_obs_features = []
        for idx in idxs:
            obs = self.obses[idx]
            next_obs = self.next_obses[idx]
            
            obs_ = obs[:, :3]
            next_obs_ = next_obs[:, :3]
            obs_feature = obs[:, 3:]
            next_obs_feature = next_obs[:, 3:]
            
            if self.config.first_subsampling_dl is not None:
                obs_subsampled, obs_feature_subsampled = grid_subsampling(obs_, features=obs_feature, sampleDl=self.config.first_subsampling_dl)
                next_obs_subsampled, next_obs_feature_subsampled = grid_subsampling(next_obs_, features=next_obs_feature, sampleDl=self.config.first_subsampling_dl)
            else:
                obs_subsampled, obs_feature_subsampled = obs_, obs_feature
                next_obs_subsampled, next_obs_feature_subsampled = next_obs_, next_obs_feature

            obses.append(obs_subsampled)
            next_obses.append(next_obs_subsampled)
            obs_features.append(obs_feature_subsampled)
            next_obs_features.append(next_obs_feature_subsampled)
        

        ###################
        # Concatenate batch
        ###################
        input_lists = []
        for tp_list in [(obses, obs_features), (next_obses, next_obs_features)]:
            points_list, features_list = tp_list
            stacked_points = np.concatenate(points_list, axis=0)
            stack_lengths = np.array([tp.shape[0] for tp in points_list], dtype=np.int32)

            # Input features
            # stacked_features_old = np.ones_like(stacked_points[:, :1], dtype=np.float32)
            stacked_features = np.concatenate(features_list, axis=0)
            # print("stacked_features_old.shape: ", stacked_features_old.shape)
            # print("stacked_features_new.shape: ", stacked_features.shape)
            # exit()

            #######################
            # Create network inputs
            #######################
            #
            #   Points, neighbors, pooling indices for each layers
            #

            # Get the whole input list
            input_list = classification_inputs(self.config, 
                                                stacked_points,
                                                stacked_features,
                                                stack_lengths)
            input_lists.append(input_list)

        obs_batch = PointCloudCustomBatch(input_lists[0])
        next_obs_batch = PointCloudCustomBatch(input_lists[1])
        obs_batch.to(self.device)
        next_obs_batch.to(self.device)
        # print(type(obs_batch))
        # print(type(next_obs_batch))
        # print(obs_batch.features)


        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        reduced_states = torch.as_tensor(self.reduced_states[idxs], device=self.device)
        return [obs_batch, reduced_states], actions, rewards, next_obs_batch, not_dones


    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __len__(self):
        return self.capacity


def classification_inputs(config,
                            stacked_points,
                            stacked_features,
                            stack_lengths):

    # Starting radius of convolutions
    if config.first_subsampling_dl is not None:
        r_normal = config.first_subsampling_dl * config.conv_radius
    else:
        r_normal = 0.033 * config.conv_radius


    # Starting layer
    layer_blocks = []

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_stack_lengths = []
    deform_layers = []

    ######################
    # Loop over the blocks
    ######################

    arch = config.architecture

    for block_i, block in enumerate(arch):

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
            layer_blocks += [block]
            continue

        # Convolution neighbors indices
        # *****************************

        deform_layer = False
        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks]):
                r = r_normal * config.deform_radius / config.conv_radius
                deform_layer = True
            else:
                r = r_normal
            conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = np.zeros((0, 1), dtype=np.int32)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
                deform_layer = True
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = np.zeros((0, 1), dtype=np.int32)
            pool_p = np.zeros((0, 1), dtype=np.float32)
            pool_b = np.zeros((0,), dtype=np.int32)

        # # Reduce size of neighbors matrices by eliminating furthest point
        # do not perform this now
        # conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
        # pool_i = self.big_neighborhood_filter(pool_i, len(input_points))

        # Updating input lists
        input_points += [stacked_points]
        input_neighbors += [conv_i.astype(np.int64)]
        input_pools += [pool_i.astype(np.int64)]
        input_stack_lengths += [stack_lengths]
        deform_layers += [deform_layer]

        # New points for next layer
        stacked_points = pool_p
        stack_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer_blocks = []

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

    ###############
    # Return inputs
    ###############

    # Save deform layers

    # list of network inputs
    li = input_points + input_neighbors + input_pools + input_stack_lengths
    li += [stacked_features]

    return li



class PointCloudCustomBatch:
    """Custom batch definition with memory pinning for ModelNet40"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        # input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 1) // 4

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        # print(self.neighbors)
        # for x in self.neighbors:
        #     print(x.dtype)
        # print(self.neighbors.dtype)
        # print(type(self.neighbors))
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])


    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        return self


def preprocess_single_obs(obs, config):
    if isinstance(obs, tuple): # handle 'rim_interpolation_and_state' mode
        obs_tmp = obs[0]
    else:
        obs_tmp = obs

    # print(type(obs_tmp))
    # exit()
    obs_ = obs_tmp[:, :3]
    feature = obs_tmp[:, 3:]
    if config.first_subsampling_dl is not None:
        obs_subsampled, feature_subsampled = grid_subsampling(obs_, features=feature, sampleDl=config.first_subsampling_dl)
    else:
        obs_subsampled, feature_subsampled = obs_, feature



    tp_list = [obs_subsampled]
    stacked_points = np.concatenate(tp_list, axis=0)
    stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
    
    # Input features
    tf_list = [feature_subsampled]
    stacked_features = np.concatenate(tf_list, axis=0)

    input_list = classification_inputs(config, 
                                        stacked_points,
                                        stacked_features,
                                        stack_lengths)

    obs_batch = PointCloudCustomBatch(input_list)

    return obs_batch