import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from upgm.utils.data_utils import get_mp4_filepaths, get_traj_hdf5_filepaths

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = False # hard-coded
        # For UPGM R2D2:
        self.mp4_filepaths = get_mp4_filepaths(data_dir=self.dataset_dir, cam_serial_num=self.camera_names[0]) # list of paths to the demonstration videos
        self.traj_hdf5_filepaths = get_traj_hdf5_filepaths(data_dir=self.dataset_dir) # list of paths to the `trajectory.h5` files containing action labels
        self.max_episode_length = 300 # hardcoded; the policy requires all sampled actions to have the same length

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        # Read an image from disk.
        hdf5_filepath = os.path.join(self.mp4_filepaths[index].split('/MP4')[0], 'images.hdf5')
        with h5py.File(hdf5_filepath, 'r') as f:
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


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=5)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=5)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


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
