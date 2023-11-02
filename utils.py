import numpy as np
import torch
import os
import h5py
import random
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from upgm.utils.data_utils import get_actions_and_target_labels_dict, get_actions_dict, get_bb_masks_hdf5_filepaths, get_images_dict, get_mp4_filepaths, get_spartn_traj_hdf5_filepaths, get_target_labels_dict, get_traj_hdf5_filepaths

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, img_size, use_moo, multiply_mask, concat_mask):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.img_size = img_size
        self.use_moo = use_moo
        self.multiply_mask = multiply_mask
        self.concat_mask = concat_mask
        # For UPGM R2D2:
        self.traj_hdf5_filepaths = get_traj_hdf5_filepaths(data_dir=self.dataset_dir) # list of paths to the `trajectory.h5` files containing action labels
        self.max_episode_length = 300 # hardcoded; the policy requires all sampled actions to have the same length
        if self.use_moo:
            self.bb_masks_hdf5_filepaths = get_bb_masks_hdf5_filepaths(data_dir=self.dataset_dir, img_size=img_size)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        # Read an image from disk.
        images_hdf5_filepath = os.path.join(self.traj_hdf5_filepaths[index].split('/trajectory.h5')[0], 'recordings', f'images_{self.img_size}px.h5')
        with h5py.File(images_hdf5_filepath, 'r') as f:
            episode_length = f['images'].shape[0]
            step_index = np.random.randint(0, episode_length)
            image = f['images'][step_index] # shape: (H, W, 3)
            image_dict = dict()
            image_dict[self.camera_names[0]] = image
        with h5py.File(self.traj_hdf5_filepaths[index], 'r') as f:
            # Read action labels from disk. Get all actions after and including step_index.
            padded_action_shape = (self.max_episode_length, f['action']['cartesian_velocity'].shape[1] + 1) # +1 on second shape dim for gripper action
            action = np.concatenate((f['action']['cartesian_velocity'][step_index:], np.expand_dims(f['action']['gripper_action'][step_index:], axis=1)), axis=1) # shape: (num_steps_after_index, action_dim)
            action = action.astype('float32') # Cast from float64 to float32. The loss of precision is negligible for our purposes.
            # Read a text annotation (e.g., 'small red cube') from disk.
            target_label = f.attrs['current_task']
            # Hard-coding joint positions/velocities as zeros since we don't use them as inputs in UPGM.
            qpos = np.zeros((7,))
            action_length = episode_length - step_index
        # If using MOO-style inputs, concatenate an object mask channel-wise to the image.
        if self.use_moo:
            bb_mask_hdf5_filepath = self.bb_masks_hdf5_filepaths[index]
            with h5py.File(bb_mask_hdf5_filepath, 'r') as f:
                if step_index >= f['bounding_box_masks'].shape[0]: # no bounding box exists
                    bb_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8) # shape: (H, W)
                else: # yes bounding box exists
                    bb_mask = f['bounding_box_masks'][step_index] # shape: (H, W)
                    # Replace the target label with the label "object". This is similar to how the object description is excluded in the original MOO paper.
                    target_label = 'object'
                bb_mask = np.expand_dims(bb_mask, -1) # shape: (H, W, 1)
                if self.multiply_mask:
                    # Multiply the image by the object mask so that all of the image except for the detected object is zero.
                    # Result: (H, W, 3) image
                    image_dict[self.camera_names[0]][:,:,:3] *= (bb_mask // 255)
                if self.concat_mask:
                    # Concatenate the object mask to the image.
                    # Result: (H, W, 4) image and mask
                    image_dict[self.camera_names[0]] = np.concatenate((image_dict[self.camera_names[0]], bb_mask), axis=-1) # shape: (H, W, 4)
        # Create padded actions tensor and is_pad bool tensors because they're required by the Transformer policy.
        padded_action = np.zeros(padded_action_shape, dtype=np.float32)
        padded_action[:action_length] = action
        is_pad = np.zeros(self.max_episode_length)
        is_pad[action_length:] = 1
        # new axis for different cameras
        # For UPGM, we only have 1 camera (the wrist camera).
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data) # This is just a way to transpose the tensor. k == # cameras == 1 in UPGM R2D2.
        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # print(f'type(image_data): {type(image_data)}')
        # print(f'type(qpos_data): {type(qpos_data)}')
        # print(f'type(action_data): {type(action_data)}')
        # print(f'type(is_pad): {type(is_pad)}')
        # print('====================')
        # print(f'image_data.shape: {image_data.shape}')
        # print(f'qpos_data.shape: {qpos_data.shape}')
        # print(f'action_data.shape: {action_data.shape}')
        # print(f'is_pad.shape: {is_pad.shape}')
        return image_data, qpos_data, action_data, is_pad, target_label


class EpisodicDatasetMemory(torch.utils.data.Dataset):
    """Data loader that loads the whole dataset into memory."""
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, img_size):
        super(EpisodicDatasetMemory).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        # For UPGM R2D2:
        mp4_filepaths = get_mp4_filepaths(data_dir=self.dataset_dir, cam_serial_num=self.camera_names[0]) # list of paths to the demonstration videos
        traj_hdf5_filepaths = get_traj_hdf5_filepaths(data_dir=self.dataset_dir) # list of paths to the `trajectory.h5` files containing action labels
        self.max_episode_length = 1000 # hardcoded; the policy requires all sampled actions to have the same length
        # Read everything from disk into memory.
        print('Loading data into memory...')
        self.images_dict = get_images_dict(mp4_filepaths, img_size)
        self.actions_and_target_labels_dict = get_actions_and_target_labels_dict(traj_hdf5_filepaths)
        print('Finished loading data.')
        # self.actions_dict = get_actions_dict(traj_hdf5_filepaths)
        # self.target_labels_dict = get_target_labels_dict(traj_hdf5_filepaths)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        # Fetch image.
        episode_length = self.images_dict[index].shape[0]
        step_index = np.random.randint(0, episode_length)
        image = self.images_dict[index][step_index] # shape: (H, W, 3)
        image_dict = dict()
        image_dict[self.camera_names[0]] = image
        # Fetch action labels. Get all actions after and including step_index.
        padded_action_shape = (self.max_episode_length, self.actions_and_target_labels_dict[index]['action'].shape[1])
        action = self.actions_and_target_labels_dict[index]['action'][step_index:] # shape: (num_steps_after_index, action_dim)
        # Read a text annotation (e.g., 'small red cube') from disk.
        target_label = self.actions_and_target_labels_dict[index]['target_label']
        # Hard-coding joint positions/velocities as zeros since we don't use them as inputs in UPGM.
        qpos = np.zeros((7,))
        action_length = episode_length - step_index
        # Create padded actions tensor and is_pad bool tensors because they're required by the Transformer policy.
        padded_action = np.zeros(padded_action_shape, dtype=np.float32)
        padded_action[:action_length] = action
        is_pad = np.zeros(self.max_episode_length)
        is_pad[action_length:] = 1
        # new axis for different cameras
        # For UPGM, we only have 1 camera (the wrist camera).
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data) # This is just a way to transpose the tensor. k == # cameras == 1 in UPGM R2D2.
        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        return image_data, qpos_data, action_data, is_pad, target_label


class AugmentedExpertedDataset(torch.utils.data.Dataset):
    """
    Apply standard data augmentation transformations to a Dataset.

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target).
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.pad = transforms.Pad(12, padding_mode='edge') # padding each side by 12 by repeating boundary pixels as in DrQ
        self.rc = transforms.RandomCrop(256, padding=0)

    def transform(self, image_data, qpos_data, action_data, is_pad, target_label):
        if random.uniform(0, 1) < 0.5: # apply augmentation 50% of the time
            images = image_data * 255.0
            # import cv2
            # cv2.imwrite('images0.png', np.transpose(images[0].numpy(), (1,2,0)))
            images = self.pad(images)
            # cv2.imwrite('images1.png', np.transpose(images[0].numpy(), (1,2,0)))
            images = self.rc(images)
            # cv2.imwrite('images2.png', np.transpose(images[0].numpy(), (1,2,0)))
            image_data = images / 255.0
        return image_data, qpos_data, action_data, is_pad, target_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(*self.dataset.__getitem__(idx))


class SPARTNDataset(torch.utils.data.Dataset):
    """
    Dataset for SPARTN augmentations.

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target).
    """
    def __init__(self, dataset_dir, camera_names, norm_stats, img_size):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.img_size = img_size
        # For UPGM R2D2:
        self.spartn_traj_hdf5_filepaths = get_spartn_traj_hdf5_filepaths(data_dir=self.dataset_dir, img_size=img_size) # list of paths to the `nerf_trajectory.h5` files containing action labels
        self.max_episode_length = 1000 # hardcoded; the policy requires all sampled actions to have the same length

    def __len__(self):
        return len(self.spartn_traj_hdf5_filepaths)

    def __getitem__(self, index):
        # Read a NeRF-augmented image from disk.
        images_hdf5_filepath = os.path.join(self.spartn_traj_hdf5_filepaths[index].split('/nerf_trajectory.h5')[0], f'nerf_images_{self.img_size}px.h5')
        with h5py.File(images_hdf5_filepath, 'r') as f:
            aug_factor = f.attrs['aug_factor'] # number of SPARTN augmentations per timestep
            aug_start_buffer = f.attrs['aug_start_buffer'] # how many timesteps in the beginning of the original episode that were ignored (not augmented via SPARTN)
            aug_index = np.random.randint(0, aug_factor) # index between 0 and aug_index
            aug_traj_len = f[f'images{aug_index}'].shape[0] # length of the partial trajectory that was augmented via SPARTN
            aug_step_index = np.random.randint(0, aug_traj_len) # timestep index between 0 and the length of the augmented trajectory
            image = f[f'images{aug_index}'][aug_step_index] # shape: (H, W, 3)
            image_dict = dict()
            image_dict[self.camera_names[0]] = image
        # Read action labels from disk. For the first timestep in the sequence, extract the action from the SPARTN corrective action
        # augmentations, but for the rest of the chunk, extract the actions from the original actions. This is because we want
        # the chunk of actions to be coherent; if we instead took a contiguous sequence of actions from the SPARTN augmentations,
        # we would have a chunk of incoherent actions corresponding to the randomly perturbed poses.
        with h5py.File(self.spartn_traj_hdf5_filepaths[index], 'r') as f:
            # Read action labels from disk. Get the actions at step_index.
            aug_action = np.append(f['action'][f'cartesian_velocity{aug_index}'][aug_step_index], f['action'][f'gripper_action{aug_index}'][aug_step_index]) # shape: (num_steps_after_index, action_dim)
            aug_action = aug_action.astype('float32') # Cast from float64 to float32. The loss of precision is negligible for our purposes.
        orig_traj_hdf5_path = os.path.join(self.spartn_traj_hdf5_filepaths[index].split(f'/nerf_{self.img_size}px')[0], 'trajectory.h5')
        step_index = aug_start_buffer + aug_step_index # offset index based on the SPARTN aug start buffer so that aug and orig timesteps align
        with h5py.File(orig_traj_hdf5_path, 'r') as f:
            # Read action labels from disk. Get all actions after the step_index.
            orig_action = np.concatenate((f['action']['cartesian_velocity'][step_index+1:], np.expand_dims(f['action']['gripper_action'][step_index+1:], axis=1)), axis=1) # shape: (num_steps_after_index, action_dim)
            orig_action = orig_action.astype('float32') # Cast from float64 to float32. The loss of precision is negligible for our purposes.
            # Read a text annotation (e.g., 'small red cube') from disk.
            target_label = f.attrs['current_task']
            # Hard-coding joint positions/velocities as zeros since we don't use them as inputs in UPGM.
            qpos = np.zeros((7,))
            episode_length = f['action']['cartesian_velocity'].shape[0]
            action_length = episode_length - step_index
            padded_action_shape = (self.max_episode_length, f['action']['cartesian_velocity'].shape[1] + 1) # +1 on second shape dim for gripper action
        # Create padded actions tensor and is_pad bool tensors because they're required by the Transformer policy.
        padded_action = np.zeros(padded_action_shape, dtype=np.float32)
        action = np.concatenate((np.expand_dims(aug_action, axis=0), orig_action), axis=0)
        padded_action[:action_length] = action
        is_pad = np.zeros(self.max_episode_length)
        is_pad[action_length:] = 1
        # new axis for different cameras
        # For UPGM, we only have 1 camera (the wrist camera).
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data) # This is just a way to transpose the tensor. k == # cameras == 1 in UPGM R2D2.
        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        return image_data, qpos_data, action_data, is_pad, target_label


def get_norm_stats(dataset_dir, num_episodes):
    traj_hdf5_filepaths = get_traj_hdf5_filepaths(dataset_dir) # list of paths to the `trajectory.h5` files containing action labels
    all_qpos_data = []
    all_action_data = []
    for hdf5_filepath in traj_hdf5_filepaths:
        with h5py.File(hdf5_filepath, 'r') as f:
            # Read action labels. Since we don't use joint positions as inputs, just set them as zeros.
            qpos = np.zeros((f['observation']['robot_state']['joint_positions'].shape[0], 7)).astype('float32') # hard-coding 7 joint positions (even though Franka has 7 + 1, where last 1 is gripper)
            action = np.concatenate((f['action']['cartesian_velocity'][:], np.expand_dims(f['action']['gripper_action'][:], axis=1)), axis=1) # shape: (num_steps_in_traj, action_dim)
            action = action.astype('float32') # Cast from float64 to float32. The loss of precision is negligible for our purposes.
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.concat(all_qpos_data)
    all_action_data = torch.concat(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10) # clipping

    # normalize qpos data
    qpos_mean = torch.zeros(qpos.shape[1]) # dummy value, just zeros
    qpos_std = torch.ones(qpos.shape[1]) # dummy value, just ones

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size, img_size, apply_aug, spartn, use_ram, use_moo, multiply_mask, concat_mask):
    print(f'\Loading data from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    if use_moo == True:
        assert use_ram == False, "use_ram==True currently not supported when use_moo==True!"
    train_dataset = EpisodicDatasetMemory(train_indices, dataset_dir, camera_names, norm_stats, img_size) if use_ram else EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, img_size, use_moo, multiply_mask, concat_mask)
    if apply_aug:
        train_dataset = AugmentedExpertedDataset(train_dataset)
    if spartn:
        spartn_dataset = SPARTNDataset(dataset_dir, camera_names, norm_stats, img_size)
        # If using SPARTN, sample from the normal training dataset and NeRF augmentations with 50% probability each.
        train_weights = len(train_dataset) * [1.0 / len(train_dataset)]
        train_weights = [0.5 * x for x in train_weights] # original dataset weights
        train_weights.extend(len(spartn_dataset) * [0.5 / len(spartn_dataset)]) # SPARTN augmentations dataset weights
        train_dataset = ConcatDataset([train_dataset, spartn_dataset])
    val_dataset = EpisodicDatasetMemory(val_indices, dataset_dir, camera_names, norm_stats, img_size) if use_ram else EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, img_size, use_moo, multiply_mask, concat_mask)
    num_workers = 0 if use_ram else len(os.sched_getaffinity(0)) # num CPU cores available to current training job -- do NOT use os.cpu_count()! -- source: https://stackoverflow.com/a/55423170
    print(f'Number of dataloader workers: {num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    print(f'\nNumber of episodes in training set: {len(train_dataset)}')
    print(f'Number of episodes in validation set: {len(val_dataset)}')
    return train_dataloader, val_dataloader, norm_stats


def load_data_debug(dataset_dir, num_episodes, camera_names, batch_size, img_size, apply_aug, spartn, use_ram, use_moo, multiply_mask, concat_mask):
    """For debugging with a tiny dataset."""
    print(f'\Loading data from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 1
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:1]
    val_indices = shuffled_indices[1:]
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    # construct dataset and dataloader
    if use_moo == True:
        assert use_ram == False, "use_ram==True currently not supported when use_moo==True!"
    train_dataset = EpisodicDatasetMemory(train_indices, dataset_dir, camera_names, norm_stats, img_size) if use_ram else EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, img_size, use_moo, multiply_mask, concat_mask)
    val_dataset = EpisodicDatasetMemory(val_indices, dataset_dir, camera_names, norm_stats, img_size) if use_ram else EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, img_size, use_moo, multiply_mask, concat_mask)
    num_workers = 0 if use_ram else len(os.sched_getaffinity(0)) # num CPU cores available to current training job -- do NOT use os.cpu_count()! -- source: https://stackoverflow.com/a/55423170
    print(f'Number of dataloader workers: {num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    print(f'\nNumber of episodes in training set: {len(train_dataset)}')
    print(f'Number of episodes in validation set: {len(val_dataset)}')
    return train_dataloader, val_dataloader, norm_stats

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def str_to_bool(s: str) -> bool:
    if s not in {'True', 'False'}:
        raise ValueError('Invalid boolean string argument given.')
    return s == 'True'
