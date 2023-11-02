import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed


class DebugPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        flattened_size = 128 + (args_override['img_size'] ** 2) * 3
        features_dim = 128
        actions_dim = 7 * args_override['num_queries'] # num_queries == chunk_size
        self.num_queries = args_override['num_queries']
        self.model = nn.Sequential(
            nn.Linear(flattened_size, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, actions_dim)
        )
        self.model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args_override['lr'])
        # TODO: Implement test-time forward pass of sentence encoder?
        # Load frozen pre-trained sentence embeddings into memory as a dict (key: target label, value: sentence embedding).
        f = open('R2D2/sentence_embeddings/sentence_embeddings_ViT-L-14@336px.json')
        data = json.load(f)
        for k in data:
            data[k] = np.array(data[k], dtype=np.float32)
        self.sentence_embeddings_dict = data
        self.sentence_encoder_proj = nn.Linear(768, 128) # project 768-d CLIP target_label embedding to 128-d embedding

    def get_frozen_sentence_embeddings(self, target_label):
        """
        Get the frozen pre-trained sentence embeddings for a list of target labels.

        Args:
            target_label: list (length = batch size) of strings
        Returns:
            Sentence embeddings with shape (batch_size, embed_size).
        """
        def label_to_embedding(label):
            return self.sentence_embeddings_dict[label]
        sentence_embeddings = list(map(label_to_embedding, target_label)) # apply `label_to_embedding` function on all elements in list
        sentence_embeddings = np.stack(sentence_embeddings) # (batch_size, D)
        sentence_embeddings = torch.Tensor(sentence_embeddings)
        sentence_embeddings = sentence_embeddings.to('cuda')
        return sentence_embeddings

    def __call__(self, qpos, image, actions=None, is_pad=None, target_label=None):
        # sentence_embeddings (1,768)
        # lang_input (1,128)
        # qpos (1,7)
        # actions (1,300,7)
        # image (1,1,3,256,256)
        # is_pad (1,300)
        env_state = None
        if actions is not None: # training time
            sentence_embeddings = self.get_frozen_sentence_embeddings(target_label)
            lang_input = self.sentence_encoder_proj(sentence_embeddings)
            image_flat = torch.flatten(image, start_dim=1)
            model_input = torch.cat((image_flat, lang_input), dim=1)
            actions = actions[:, :self.num_queries] # (batch_size, seq_len, action_dim)
            a_hat = self.model(model_input)
            a_hat = torch.reshape(a_hat, actions.shape)
            is_pad = is_pad[:, :self.num_queries] # (batch_size, seq_len)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none') # (batch_size, seq_len, action_dim)
            all_l1_dxyz = F.l1_loss(actions[:,:,:3], a_hat[:,:,:3], reduction='none') # (batch_size, seq_len, 3)
            all_l1_dEuler = F.l1_loss(actions[:,:,3:6], a_hat[:,:,3:6], reduction='none') # (batch_size, seq_len, 3)
            all_l1_dgrip = F.l1_loss(actions[:,:,6], a_hat[:,:,6], reduction='none').unsqueeze(-1) # (batch_size, seq_len, 1)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean() # scalar value
            l1_dxyz = (all_l1_dxyz * ~is_pad.unsqueeze(-1)).mean() # scalar value
            l1_dEuler = (all_l1_dEuler * ~is_pad.unsqueeze(-1)).mean() # scalar value
            l1_dgrip = (all_l1_dgrip * ~is_pad.unsqueeze(-1)).mean() # scalar value
            loss_dict['l1'] = l1
            loss_dict['l1_dxyz'] = l1_dxyz
            loss_dict['l1_dEuler'] = l1_dEuler
            loss_dict['l1_dgrip'] = l1_dgrip
            loss_dict['loss'] = loss_dict['l1']
            print(f'actions: {actions}') # TODO
            print(f'a_hat: {a_hat}') # TODO

            return loss_dict
        else: # inference time
            a_hat, _, = self.model(qpos, image, env_state, target_label=target_label) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None, target_label=None):
        env_state = None
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries] # (batch_size, seq_len, action_dim)
            is_pad = is_pad[:, :self.model.num_queries] # (batch_size, seq_len)
            a_hat, is_pad_hat = self.model(qpos, image, env_state, actions, is_pad, target_label)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none') # (batch_size, seq_len, action_dim)
            all_l1_dxyz = F.l1_loss(actions[:,:,:3], a_hat[:,:,:3], reduction='none') # (batch_size, seq_len, 3)
            all_l1_dEuler = F.l1_loss(actions[:,:,3:6], a_hat[:,:,3:6], reduction='none') # (batch_size, seq_len, 3)
            all_l1_dgrip = F.l1_loss(actions[:,:,6], a_hat[:,:,6], reduction='none').unsqueeze(-1) # (batch_size, seq_len, 1)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean() # scalar value
            l1_dxyz = (all_l1_dxyz * ~is_pad.unsqueeze(-1)).mean() # scalar value
            l1_dEuler = (all_l1_dEuler * ~is_pad.unsqueeze(-1)).mean() # scalar value
            l1_dgrip = (all_l1_dgrip * ~is_pad.unsqueeze(-1)).mean() # scalar value
            loss_dict['l1'] = l1
            loss_dict['l1_dxyz'] = l1_dxyz
            loss_dict['l1_dEuler'] = l1_dEuler
            loss_dict['l1_dgrip'] = l1_dgrip
            loss_dict['loss'] = loss_dict['l1']
            return loss_dict
        else: # inference time
            a_hat, _, = self.model(qpos, image, env_state, target_label=target_label) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
