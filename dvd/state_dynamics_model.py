from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/home/akannan2/rewards-from-human-videos/handful-of-trials-pytorch')
from config.utils import swish, get_affine_params

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from tqdm import trange

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NPART = 20 # number of particles
EPOCHS = 5 # model_train_cfg['epochs']

class Dataset:
    def __init__(self, nx, nu, maxsize=10000):
        self.nx = nx
        self.nu = nu
        self.maxsize = maxsize
        self.initialized = False

        self.inputs = None
        self.targets = None

    def add(self, obs_traj, acs_traj):
        """
        :param obs_traj: np.ndarray
            Observations of shape (T+1) x nx
        :param acs_traj: np.ndarray 
            Actions of shape T x nu
        """
        # import ipdb; ipdb.set_trace()
        if not self.initialized:
            assert len(obs_traj.shape) == 2 and obs_traj.shape[1] == self.nx
            assert len(acs_traj.shape) == 2 and acs_traj.shape[1] == self.nu
            assert obs_traj.shape[0] == acs_traj.shape[0] + 1

            states = obs_traj[:-1]
            next_states = obs_traj[1:]
            new_inputs = np.concatenate([states, acs_traj], axis=1)
            new_targets = next_states - states

            self.inputs = new_inputs
            self.targets = new_targets
            self.initialized = True

        else:
            states = obs_traj[:-1]
            next_states = obs_traj[1:]
            new_inputs = np.concatenate([states, acs_traj], axis=1)
            new_targets = next_states - states

            self.inputs = np.concatenate([self.inputs, new_inputs], axis=0)
            self.targets = np.concatenate([self.targets, new_targets], axis=0)

        if self.inputs.shape[0] > self.maxsize:
            idxs = np.random.choice(self.inputs.shape[0], self.maxsize, replace=False)
            self.inputs = self.inputs[idxs]
            self.targets = self.targets[idxs]

    def sample(self, num_samples):
        """
        :param num_samples int
        :returns random sample of inputs and targets 
        """
        idxs = np.random.choice(self.inputs.shape[0], num_samples, replace=False)
        return self.inputs[idxs], self.targets[idxs]

    def get_inputs_targets(self):
        return self.inputs, self.targets

class PtModel(nn.Module):

    def __init__(self, ensemble_size, in_features, out_features):
        super().__init__()

        self.num_nets = ensemble_size

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, 200)

        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, 200, 200)

        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, 200, 200)

        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, 200, out_features)

        self.inputs_mu = nn.Parameter(torch.zeros(in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(in_features), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32) * 10.0)

    def compute_decays(self):

        lin0_decays = 0.0025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.005 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.0075 * (self.lin3_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):
        inputs = inputs.to(TORCH_DEVICE)

        # Transform inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)

def nn_constructor():
    ensemble_size = 5
    nx = 13 # state space size
    nu = 4 # action space size

    model_in = nx + nu # given state and action
    model_out = nx * 2 # predict next state mean and next state variance

    model = PtModel(ensemble_size, model_in, model_out).to(TORCH_DEVICE)
    model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

    return model

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]

def train(model, dataset, obs_trajs=None, acs_trajs=None):
    """
    :param model PtModel
    :param dataset Dataset
    :param obs_traj: List[np.ndarray]
        Observations of shape B x (T+1) x nx
    :param acs_traj: List[np.ndarray]
        Actions of shape B x T x nu
    """
    batch_size = 32

    # Construct new training points and add to training set
    if obs_trajs is not None and acs_trajs is not None:
        for obs, acs in zip(obs_trajs, acs_trajs):
            dataset.add(obs, acs)

    # Train the model
    # self.has_been_trained = True

    # Train the pytorch model
    inputs, targets = dataset.get_inputs_targets()
    # print('mu', model.inputs_mu.data)
    # print('sigma', model.inputs_sigma.data)
    # for k, v in model.state_dict().items():
    #     print(k, v.min(), v.max())
    model.fit_input_stats(inputs)

    idxs = np.random.randint(inputs.shape[0], size=[model.num_nets, inputs.shape[0]])

    epoch_range = trange(EPOCHS, unit="epoch(s)", desc="Network training")
    num_batch = int(np.ceil(idxs.shape[-1] / batch_size))
    mse_losses_list = np.zeros(EPOCHS)

    for ep in epoch_range:

        for batch_num in range(num_batch):
            batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]

            loss = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
            loss += model.compute_decays()

            # TODO: move all training data to GPU before hand
            train_in = torch.from_numpy(inputs[batch_idxs]).to(TORCH_DEVICE).float()
            train_targ = torch.from_numpy(targets[batch_idxs]).to(TORCH_DEVICE).float()

            # import ipdb; ipdb.set_trace()

            mean, logvar = model(train_in, ret_logvar=True)
            inv_var = torch.exp(-logvar)

            train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
            train_losses = train_losses.mean(-1).mean(-1).sum()
            # Only taking mean over the last 2 dimensions
            # The first dimension corresponds to each model in the ensemble

            loss += train_losses

            model.optim.zero_grad()
            loss.backward()
            model.optim.step()

        idxs = shuffle_rows(idxs)

        # TODO: not a "validation" set but we can record real val loss later
        val_in = torch.from_numpy(inputs[idxs[:5000]]).to(TORCH_DEVICE).float()
        val_targ = torch.from_numpy(targets[idxs[:5000]]).to(TORCH_DEVICE).float()

        mean, _ = model(val_in)
        mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)

        epoch_range.set_postfix({
            "Training loss(es)": mse_losses.detach().cpu().numpy()
        })
        mse_losses_list[ep] = mse_losses.mean().detach().cpu().item()

    return mse_losses_list.mean()

@torch.no_grad()
def rollout_trajectory(init_state, ac_seqs, model):
    """
    :param init_state np.ndarray (nx,)
    :param ac_seqs torch.Tensor (T, nu)
    """
    ac_seqs = ac_seqs.float().to(TORCH_DEVICE)
    expanded = ac_seqs.unsqueeze(1)
    ac_seqs = expanded.expand(-1, NPART, -1) # T x NPART x nu

    cur_obs = torch.from_numpy(init_state).float().to(TORCH_DEVICE)
    cur_obs = cur_obs[None]
    cur_obs = cur_obs.expand(NPART, -1) # NPART x nx

    all_obs = []
    for t in range(ac_seqs.shape[0]):
        cur_acs = ac_seqs[t]
        cur_obs = _predict_next_obs(cur_obs, cur_acs, model)
        all_obs.append(cur_obs.cpu().numpy())

    # print(t, ':', torch.isnan(cur_obs.view(-1)).sum().item())

    return np.array(all_obs) # cur_obs.cpu().numpy()

def _predict_next_obs(obs, acs, model):
    proc_obs = obs # self.obs_preproc(obs)

    # assert self.prop_mode == 'TSinf'

    proc_obs = _expand_to_ts_format(proc_obs, model)
    acs = _expand_to_ts_format(acs, model)

    inputs = torch.cat((proc_obs, acs), dim=-1)

    mean, var = model(inputs)

    predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()

    # TS Optimization: Remove additional dimension
    predictions = _flatten_to_matrix(predictions, model)

    return obs + predictions

def _expand_to_ts_format(mat, model):
    dim = mat.shape[-1]

    # Before, [10, 5] in case of proc_obs
    reshaped = mat.view(-1, model.num_nets, NPART // model.num_nets, dim)
    # After, [2, 5, 1, 5]

    transposed = reshaped.transpose(0, 1)
    # After, [5, 2, 1, 5]

    reshaped = transposed.contiguous().view(model.num_nets, -1, dim)
    # After. [5, 2, 5]

    return reshaped

def _flatten_to_matrix(ts_fmt_arr, model):
    dim = ts_fmt_arr.shape[-1]

    reshaped = ts_fmt_arr.view(model.num_nets, -1, NPART // model.num_nets, dim)

    transposed = reshaped.transpose(0, 1)

    reshaped = transposed.contiguous().view(-1, dim)

    return reshaped
