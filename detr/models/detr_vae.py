# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import clip
import json
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, sentence_encoder, state_dim, num_queries, camera_names, sentence_embeddings_path=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            sentence_encoder: Sentence encoder (e.g., CLIP language encoder or DistilBERT).
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            camera_names: Names of cameras used for data collection and training.
            sentence_embeddings_path: (Optional) Path to frozen sentence embeddings.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        # At training time, we want to load frozen pre-trained sentence embeddings instead of running unnecessary forward passes
        # through the large sentence encoder model -- since the target labels are fixed anyway. Therefore, there is no need to
        # initialize the sentence encoder at all. We just need to load saved sentence embeddings.
        # At test time, the user might enter an unseen target label, so we do need to run a forward pass through the sentence encoder.
        if sentence_embeddings_path is not None: # train time
            self.sentence_embeddings_path = sentence_embeddings_path
            self.use_frozen_sentence_embeddings = True
            # Load frozen pre-trained sentence embeddings into memory as a dict (key: target label, value: sentence embedding).
            f = open(sentence_embeddings_path)
            data = json.load(f)
            for k in data:
                data[k] = np.array(data[k], dtype=np.float32)
            self.sentence_embeddings_dict = data
        else: # test time
            self.sentence_encoder = sentence_encoder
            self.use_frozen_sentence_embeddings = False
        hidden_dim = transformer.d_model
        self.sentence_encoder_proj = nn.Linear(768, hidden_dim) # project 768-d CLIP target_label embedding to 512-d Transformer embedding
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(7, hidden_dim)
        else:
            # input_dim = 7 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(7, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
        # decoder extra parameters
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and language embedding

    def get_frozen_sentence_embeddings(self, target_label):
        """
        Get the frozen pre-trained sentence embeddings for a list of target labels.

        Args:
            target_label: list (length = batch size) of strings
        Returns:
            Sentence embeddings with shape (batch_size, embed_size).
        """
        assert self.use_frozen_sentence_embeddings, "Error: Tried to retrieve frozen sentence embeddings, but self.use_frozen_sentence_embeddings == False."
        def label_to_embedding(label):
            return self.sentence_embeddings_dict[label]
        sentence_embeddings = list(map(label_to_embedding, target_label)) # apply `label_to_embedding` function on all elements in list
        sentence_embeddings = np.stack(sentence_embeddings) # (batch_size, D)
        sentence_embeddings = torch.Tensor(sentence_embeddings)
        sentence_embeddings = sentence_embeddings.to('cuda')
        return sentence_embeddings

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, target_label=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        target_label: list (length = batch size) of strings
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            # Sentence encoder
            # - If we are training, load frozen sentence embeddings.
            # - Else, do a forward pass through the sentence encoder.
            if self.use_frozen_sentence_embeddings:
                sentence_embeddings = self.get_frozen_sentence_embeddings(target_label)
            else:
                with torch.no_grad():
                    tokens = clip.tokenize(target_label).to('cuda')
                    sentence_embeddings = self.sentence_encoder.encode_text(tokens).float()
            lang_input = self.sentence_encoder_proj(sentence_embeddings)
            hs = self.transformer(src, None, self.query_embed.weight, pos, proprio_input, self.additional_pos_embed.weight, lang_input)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 7
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=7, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 7
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_sentence_encoder():
    sentence_encoder, _ = clip.load("ViT-L/14@336px", device='cuda') # ViT-L/14@336px has highest performance among CLIP models; embedding size == 768
    for param in sentence_encoder.parameters():
        param.requires_grad = False
    return sentence_encoder


def build(args):
    state_dim = 7 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    sentence_encoder = build_sentence_encoder()

    # Don't use frozen pre-trained sentence at test time, i.e., actually run a forward pass through the sentence encoder.
    if args.eval:
        args.sentence_embeddings_path = None

    model = DETRVAE(
        backbones,
        transformer,
        sentence_encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        sentence_embeddings_path=args.sentence_embeddings_path,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 7 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

