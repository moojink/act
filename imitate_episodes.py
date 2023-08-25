import cv2
import imageio
import json
import torch
import time
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from utils import str_to_bool # for arg parsing bool args
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from r2d2.robot_env import RobotEnv
from torch.utils.tensorboard import SummaryWriter
from upgm.utils.data_utils import get_mp4_filepaths, get_traj_hdf5_filepaths, read_mp4
from dummy_robot_env import DummyRobotEnv # TODO remove

import IPython
e = IPython.embed

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main(args):
    set_seed(1)
    # command line parameters
    if not args.eval:
        args.log_dir = update_log_dir(args.log_dir)

    # get task parameters
    if not args.eval:
        dataset_dir = args.data_dir
        mp4_filepaths = get_mp4_filepaths(data_dir=dataset_dir, cam_serial_num=args.cam_serial_num) # list of paths to the demonstration videos
        num_episodes = len(mp4_filepaths)
    episode_len = 100
    camera_names = [args.cam_serial_num]

    # fixed parameters
    state_dim = 7 # this is like the action dim
    if args.policy_class == 'ACT':
        policy_config = {'lr': args.lr,
                         'num_queries': args.chunk_size,
                         'kl_weight': args.kl_weight,
                         'hidden_dim': args.hidden_dim,
                         'dim_feedforward': args.dim_feedforward,
                         'lr_backbone': args.lr,
                         'backbone': args.image_encoder,
                         'enc_layers': args.enc_layers,
                         'dec_layers': args.dec_layers,
                         'nheads': args.nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args.lr, 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    if args.eval:
        success_rate, avg_return = eval_bc(args, policy_config, save_episode=True)
        exit()

    train_dataloader, val_dataloader, stats = load_data(dataset_dir, num_episodes, camera_names, args.batch_size, args.img_size, args.apply_aug, args.spartn, args.use_ram)

    # save dataset stats
    stats_path = os.path.join(args.log_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Save a json file with the bc_args used for this run (for reference).
    with open(os.path.join(args.log_dir, 'bc_args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, args, policy_config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    if not args.debug:
        checkpoint_path = os.path.join(args.log_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, checkpoint_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
        num_total_params = sum(p.numel() for p in policy.parameters())
        print(f'Total # parameters: {num_total_params}')
        num_trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f'# trainable parameters: {num_trainable_params}\n')
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def save_rollout_gif(img_list):
    """Turns a list of images into a video and saves it."""
    if img_list == []:
        return
    speedup = 3
    img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]
    rollout_path = 'rollout.gif'
    effective_fps = 15 * speedup
    duration_in_milliseconds = 1000 * 1 / effective_fps
    imageio.mimwrite(rollout_path, img_list, duration=duration_in_milliseconds, loop=True)
    print(f'Saved rollout GIF at path {rollout_path}')

def eval_bc(args, policy_config, save_episode=True):
    set_seed(args.seed)
    max_timesteps = 600

    # load policy and stats
    checkpoint_path = os.path.join(args.checkpoint_dir, f'policy_epoch_{args.checkpoint_epoch}_seed_{args.seed}.ckpt')
    policy = make_policy(args.policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(checkpoint_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {checkpoint_path}')
    stats_path = os.path.join(args.checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # Make the robot environment.
    env = RobotEnv(action_space='cartesian_velocity')
    env_max_reward = 0
    query_frequency = policy_config['num_queries']
    if args.temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    target_label = ''

    while True:
        # Ask the user for the target object description.
        if target_label == '':
            target_label = input('Enter the target object to grasp: ')
        else:
            user_input = input('Enter the target object to grasp. To repeat the previous target object, press Enter without typing anything: ')
            if user_input != '':
                target_label = user_input
        print(f'Target object to grasp: {target_label}')
        env.reset()
        input('Press Enter to begin...')


        ### evaluation loop
        if args.temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, policy_config['state_dim']]).cuda()
        qpos_history = torch.zeros((1, max_timesteps, 7)).cuda()
        image_list = [] # for visualization
        with torch.inference_mode():
            for t in range(max_timesteps):
                try:
                    print(f'step: {t}')
                    # Mark starting time.
                    step_start_time = time.time()
                    ### process previous timestep to get qpos and image_list
                    # Get environment image observations.
                    obs_dict = env.get_observation()
                    image = obs_dict['image'][policy_config['camera_names'][0]][:] # shape: (480, 640, 3)
                    image = image[:, 80:560, :] # shape: (480, 480, 3)
                    image = cv2.resize(image, (256, 256)) # shape: (256, 256, 3)
                    image_list.append(image)
                    # Get robot state.
                    # For now, we just set the robot state to all zeros since the policy does not have proprioceptive inputs.
                    # qpos = np.append(obs_dict['robot_state']['joint_positions'], obs_dict['robot_state']['gripper_position'])
                    qpos = np.zeros((7,))
                    qpos = pre_process(qpos)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos
                    # Reshape the image before feeding it into the policy.
                    image = torch.from_numpy(image).float().cuda().unsqueeze(0) # shape: (num_cameras=1, H, W, C)
                    image = image.unsqueeze(0) # shape: (batch_size=1, num_cameras=1, H, W, C)
                    image = torch.einsum('n k h w c -> n k c h w', image) # shape: (batch_size=1, num_cameras=1, C, H, W)
                    # normalize image and change dtype to float
                    image = image / 255.0
                    ### query policy
                    if args.policy_class == "ACT":
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, image, target_label=target_label)
                        if args.temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif args.policy_class == "CNNMLP":
                        raw_action = policy(qpos, image, target_label=target_label)
                    else:
                        raise NotImplementedError
                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    action = np.clip(action, -1, 1)
                    ### step the environment
                    print(f'action: {action}')
                    action_info = env.step(action)
                    # Sleep the amount necessary to maintain consistent control frequency.
                    elapsed_time = time.time() - step_start_time
                    time_to_sleep = (1 / env.control_hz) - elapsed_time
                    print('time_to_sleep:', time_to_sleep)
                    if time_to_sleep > 0:
                        time.sleep(time_to_sleep)
                except KeyboardInterrupt:
                    # save_rollout_gif(image_list)
                    image_list = [] # reset the episode replay GIF
                    user_input = input('\nEnter (Ctrl-C) to quit the program, or anything else to continue to the next episode...')
                    break


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, target_label = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad, target_label) # TODO remove None


def update_log_dir(log_dir):
    """
    Updates the log directory by appending a new experiment number.
    Example: 'logs/bc/' -> 'logs/bc/1' if 'logs/bc/0' exists and is the only experiment completed thus far.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sub_dirs = [sub_dir for sub_dir in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, sub_dir))]
    if len(sub_dirs) == 0:
        log_dir = os.path.join(log_dir, '0')
    else:
        sub_dirs_as_ints = [int(s) for s in sub_dirs]
        last_sub_dir = max(sub_dirs_as_ints)
        log_dir = os.path.join(log_dir, str(last_sub_dir + 1))
    os.makedirs(log_dir)
    return log_dir


def train_bc(train_dataloader, val_dataloader, args, policy_config):
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    print(f'\nLogging to directory {args.log_dir}\n')

    set_seed(args.seed)

    policy = make_policy(args.policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(args.policy_class, policy)

    # Load checkpoint if applicable.
    if args.checkpoint_epoch != '':
        checkpoint_path = os.path.join(args.checkpoint_dir, f'policy_epoch_{args.checkpoint_epoch}_seed_{args.seed}.ckpt')
        checkpoint = torch.load(checkpoint_path)
        policy.load_state_dict(checkpoint, strict=False)
        print(f'Loaded checkpoint from {checkpoint_path}')
        # Load optimizer state if applicable.
        if args.load_optimizer:
            optimizer_state_path = os.path.join(args.checkpoint_dir, f'optimizer_epoch_{args.checkpoint_epoch}_seed_{args.seed}.ckpt')
            optimizer_state = torch.load(optimizer_state_path)
            optimizer.load_state_dict(optimizer_state)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    # Offset start and end epoch numbers if args.checkpoint_epoch_offset==True so that we start where we
    # left off in the previous training run.
    epoch_start = 1
    epoch_end = args.num_epochs
    if args.checkpoint_epoch != '' and args.checkpoint_epoch_offset==True:
        epoch_start = int(args.checkpoint_epoch) + 1
        epoch_end += int(args.checkpoint_epoch)
    for epoch in tqdm(range(epoch_start, epoch_end + 1)):
        print(f'\nEpoch {epoch}')
        # validation
        start_time = time.time()
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            epoch_val_loss_l1 = epoch_summary['l1']
            epoch_val_loss_l1_dxyz = epoch_summary['l1_dxyz']
            epoch_val_loss_l1_dEuler = epoch_summary['l1_dEuler']
            epoch_val_loss_l1_dgrip = epoch_summary['l1_dgrip']
            epoch_val_loss_kl = epoch_summary['kl'] * args.kl_weight
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        elapsed_time = time.time() - start_time
        print(f'Val loss:   {epoch_val_loss:.5f}')
        print(f'Val loss (L1):   {epoch_val_loss_l1:.5f}')
        print(f'Val loss_dxyz (L1):   {epoch_val_loss_l1_dxyz:.5f}')
        print(f'Val loss_dEuler (L1):   {epoch_val_loss_l1_dEuler:.5f}')
        print(f'Val loss_dgrip (L1):   {epoch_val_loss_l1_dgrip:.5f}')
        print(f'Val loss (KL):   {epoch_val_loss_kl:.5f}')
        print(f'Seconds per epoch (val):   {elapsed_time:.5f}')
        if 0 <= epoch <= 1 or epoch % args.tb_writer_interval == 0:
            tb_writer.add_scalar(f'loss (val)', epoch_val_loss, epoch)
            tb_writer.add_scalar(f'loss L1 (val)', epoch_val_loss_l1, epoch)
            tb_writer.add_scalar(f'loss KL (val)', epoch_val_loss_kl, epoch)
            tb_writer.add_scalar(f'sec/epoch (val)', elapsed_time, epoch)
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        epoch_dicts = []
        start_time = time.time()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_dicts.append(detach_dict(forward_dict))
        elapsed_time = time.time() - start_time
        epoch_summary = compute_dict_mean(epoch_dicts)
        train_history.append(epoch_summary)
        epoch_train_loss = epoch_summary['loss']
        epoch_train_loss_l1 = epoch_summary['l1']
        epoch_train_loss_l1_dxyz = epoch_summary['l1_dxyz']
        epoch_train_loss_l1_dEuler = epoch_summary['l1_dEuler']
        epoch_train_loss_l1_dgrip = epoch_summary['l1_dgrip']
        epoch_train_loss_kl = epoch_summary['kl'] * args.kl_weight
        print(f'Train loss: {epoch_train_loss:.5f}')
        print(f'Train loss (L1): {epoch_train_loss_l1:.5f}')
        print(f'Train loss_dxyz (L1): {epoch_train_loss_l1_dxyz:.5f}')
        print(f'Train loss_dEuler (L1): {epoch_train_loss_l1_dEuler:.5f}')
        print(f'Train loss_dgrip (L1): {epoch_train_loss_l1_dgrip:.5f}')
        print(f'Train loss (KL): {epoch_train_loss_kl:.5f}')
        print(f'Seconds per epoch (train):   {elapsed_time:.5f}')
        if 0 <= epoch <= 1 or epoch % args.tb_writer_interval == 0:
            tb_writer.add_scalar(f'loss (train)', epoch_train_loss, epoch)
            tb_writer.add_scalar(f'loss L1 (train)', epoch_train_loss_l1, epoch)
            tb_writer.add_scalar(f'loss KL (train)', epoch_train_loss_kl, epoch)
            tb_writer.add_scalar(f'sec/epoch (train)', elapsed_time, epoch)
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % int(args.num_epochs / 10) == 0 and not args.debug:
            checkpoint_path = os.path.join(args.log_dir, f'policy_epoch_{epoch}_seed_{args.seed}.ckpt')
            torch.save(policy.state_dict(), checkpoint_path)

    # Save final epoch checkpoint with optimizer state as well so that we can later resume training from where we left off.
    if not args.debug:
        checkpoint_path = os.path.join(args.log_dir, f'policy_epoch_{epoch_end}_seed_{args.seed}.ckpt')
        torch.save(policy.state_dict(), checkpoint_path)
        optimizer_state_path = os.path.join(args.log_dir, f'optimizer_epoch_{epoch_end}_seed_{args.seed}.ckpt')
        torch.save(optimizer.state_dict(), optimizer_state_path)

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        checkpoint_path = os.path.join(args.log_dir, f'policy_epoch_{best_epoch}_seed_{args.seed}.ckpt')
        torch.save(best_state_dict, checkpoint_path)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, log_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(log_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {log_dir}')


def str_to_bool(s: str) -> bool:
    if s not in {'True', 'False'}:
        raise ValueError('Invalid boolean string argument given.')
    return s == 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument("--log_dir", type=str, default='act/checkpoints', help="Logs directory for TensorBoard stats and policy demo gifs.")
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per gradient step.")
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument("--num_epochs", type=int, default=100000, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=float, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8)
    parser.add_argument('--temporal_agg', action='store_true')

    # from UPGM R2D2
    parser.add_argument("--data_dir", type=str, default='R2D2/data',
                        help="Directory containing the expert demonstrations used for training.")
    parser.add_argument("--cam_serial_num", type=str, default='138422074005',
                        help="Serial number of the camera used to record videos of the demonstration trajectories.")
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory containing the saved checkpoint.")
    parser.add_argument("--checkpoint_epoch", type=str, default='',
                        help="The epoch number at which to resume training. If 0, start fresh.")
    parser.add_argument("--load_optimizer", type=str_to_bool, default=False,
                        help="(Only applicable when loading checkpoint) Whether to load the previously saved optimizer state.")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Size of (square) image observations.")
    parser.add_argument("--image_encoder", type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'],
                        help="Which image encoder to use for the BC policy.")
    parser.add_argument("--sentence_embeddings_path", type=str, default='R2D2/sentence_embeddings/sentence_embeddings_ViT-L-14@336px.json',
                        help="Path to frozen sentence embeddings (optional, for faster training).")
    parser.add_argument("--apply_aug", type=str_to_bool, default=False,
                        help="Whether to use standard data augmentations on the training set (e.g., random crop).")
    parser.add_argument("--spartn", type=str_to_bool, default=False,
                        help="Whether to use SPARTN data augmentations on the training set.")
    parser.add_argument("--use_ram", type=str_to_bool, default=False,
                        help="Whether to load all training data into memory instead of reading from disk (for small datasets).")
    parser.add_argument("--checkpoint_epoch_offset", type=str_to_bool, default=False,
                        help="(Only applicable when loading checkpoint) If True, the starting epoch number is 0. Else, we start where the previous checkpoint finished.")
    parser.add_argument("--tb_writer_interval", type=int, default=100,
                        help="We write to TensorBoard once per `tb_writer_interval` steps.")
    parser.add_argument("--debug", type=str_to_bool, default=False,
                        help="Whether to enable debugging mode.")

    args = parser.parse_args()
    main(args)
