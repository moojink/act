import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, target_label=None):
        env_state = None
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries] # (batch_size, seq_len, action_dim)
            is_pad = is_pad[:, :self.model.num_queries] # (batch_size, seq_len)
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad, target_label)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
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
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, target_label=target_label) # no action, sample from prior
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

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
