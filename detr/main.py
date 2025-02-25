# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model
from .util.misc import str_to_bool # for arg parsing bool args

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument("--log_dir", type=str, default='act/checkpoints', help="Logs directory for TensorBoard stats and policy demo gifs.")
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument("--num_epochs", type=int, default=100000, help="Number of epochs to train for.")
    parser.add_argument('--kl_weight', action='store', type=float, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
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
    parser.add_argument("--image_encoder", type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b0film', 'efficientnet_b3film'],
                        help="Which image encoder to use for the BC policy.")
    parser.add_argument("--sentence_embeddings_path", type=str, default='R2D2/sentence_embeddings/sentence_embeddings_ViT-L-14@336px.json',
                        help="Path to frozen sentence embeddings (optional, for faster training).")
    parser.add_argument("--apply_aug", type=str_to_bool, default=True,
                        help="Whether to apply data augmentations on the training set (e.g., random crop).")
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
    parser.add_argument("--use_moo", type=str_to_bool, default=False,
                        help="Whether to enable MOO-style object detection.")
    parser.add_argument("--multiply_mask", type=str_to_bool, default=False,
                        help="(Only applicable when --use_moo==True) Whether to multiply the RGB image by the object mask output by OWL-ViT object detector.")
    parser.add_argument("--concat_mask", type=str_to_bool, default=True,
                        help="(Only applicable when --use_moo==True) Whether to concatenate the RGB image and object mask output by OWL-ViT object detector.")

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

