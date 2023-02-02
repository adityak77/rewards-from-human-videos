from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

import sys
sys.path.append('/home/akannan2/rewards-from-human-videos/pydreamer')
from pydreamer.models.dreamer import Dreamer
from pydreamer.models.functions import map_structure

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@torch.no_grad()
def rollout_trajectory(init_state, init_actions, ac_seqs, world_model):
    """
    :param init_state np.ndarray (T_past, B, C, H, W) - states should be same across batch
    :param init_actions np.ndarray (T_past, B, nu)
    :param ac_seqs torch.Tensor (B, T, nu)

    :return all_obs np.ndarray (B, T, H, W, C)
    """
    T_past, B, C, H, W = init_state.shape
    T = ac_seqs.shape[1]
    iwae_samples = 1
    do_open_loop = False

    ac_seqs = ac_seqs.float().to(TORCH_DEVICE)
    init_state = torch.tensor(init_state, dtype=torch.float32, device=TORCH_DEVICE)
    init_actions = torch.tensor(init_actions, dtype=torch.float32, device=TORCH_DEVICE)

    all_obs = []
    all_features = []
    
    obs = {'image': init_state,
           'action' : init_actions,
           'reset': torch.zeros((T_past, B), dtype=torch.bool, device=TORCH_DEVICE),
          } # first input
    in_state = world_model.init_state(B * iwae_samples) # first input
    
    with torch.no_grad():
        embed = world_model.encoder(obs) # only obs['image'] used here
        _, _, _, _, states, _ = \
            world_model.core.forward(embed,
                                    obs['action'],
                                    obs['reset'],
                                    in_state,
                                    iwae_samples=iwae_samples,
                                    do_open_loop=do_open_loop)

        state = map_structure(states, lambda x: x.detach()[-1, :, 0])
        for t in range(ac_seqs.shape[1]):
            feature = world_model.core.to_feature(*state)
            cur_ac = ac_seqs[:, t]
            
            if t > 0:
                all_features.append(feature)
            _, state = world_model.core.cell.forward_prior(cur_ac, None, state)

        feature = world_model.core.to_feature(*state)
        all_features.append(feature)
        all_features = torch.stack(all_features)

        all_obs = world_model.decoder.image.forward(all_features)

    all_obs = all_obs.detach().cpu().numpy()

    return all_obs.transpose(1, 0, 3, 4, 2) # should be B x T x H x W x C


def load_world_model(conf, model_weights_path):
    assert conf.model == 'dreamer'
    model = Dreamer(conf)

    model.load_state_dict(torch.load(model_weights_path)['model_state_dict'])
    world_model = model.wm
    return world_model.to(TORCH_DEVICE).eval()
