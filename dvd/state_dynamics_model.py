from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mimetypes import init

import sys
sys.path.append('/home/akannan2/rewards-from-human-videos/handful-of-trials-pytorch')
from config.utils import swish, get_affine_params

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Dataset():
    def __init__(self, nx, nu, ac_lb, ac_ub):
        self.nx = nx
        self.nu = nu
        self.ac_lb = ac_lb
        self.ac_ub = ac_ub
        self.initialized = False
        self.inputs = None
        self.targets = None

    def add(self, obs, actions):
        """
        obs: np.ndarray (T+1) x nx
        actions: np.ndarray T x nu
        """
        assert len(obs.shape) == 2 and len(actions.shape) == 2
        assert obs.shape[0] == actions.shape[0] + 1
        assert obs.shape[1] == self.nx and actions.shape[1] == self.nu

        s = obs[:-1]
        s_next = obs[1:]
        actions = np.clip(actions, self.ac_lb, self.ac_ub)

        new_inputs = np.concatenate((s, actions), axis=1)
        new_targets = s_next - s

        if not self.initialized:
            self.inputs = new_inputs
            self.targets = new_targets
            self.initialized = True

        else:
            self.inputs = np.concatenate((self.inputs, new_inputs), axis=0)
            self.targets = np.concatenate((self.targets, new_targets), axis=0)

    def sample(self, bsize):
        inds = np.random.choice(len(self.inputs), bsize, replace=False) # replace=True
        inputs_batch = self.inputs[inds]
        targets_batch = self.targets[inds]

        return torch.tensor(inputs_batch, device=TORCH_DEVICE), torch.tensor(targets_batch, device=TORCH_DEVICE)

    def get_inputs_targets(self):
        return torch.tensor(self.inputs, device=TORCH_DEVICE), torch.tensor(self.targets, device=TORCH_DEVICE)

# vanilla dynamics model
class DynamicsModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_features, out_features)
        ).double().to(TORCH_DEVICE)

    def forward(self, x):
        x = x.double().to(TORCH_DEVICE)
        return self.network(x)

def get_vanilla_dynamics_model():
    nx = 13
    nu = 4
    hidden = 200

    model = DynamicsModel(nx+nu, nx, hidden)
    model.optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model

def train(model, dataset):
    BATCH_SIZE = 32
    TRAIN_EPOCHS = 150

    total_loss = 0
    for epoch in range(TRAIN_EPOCHS):
        inputs, targets = dataset.sample(BATCH_SIZE)
        preds = model(inputs)
        
        model.optim.zero_grad()
        loss = ((preds - targets).norm(2, dim=1) ** 2).mean()
        loss.backward()
        model.optim.step()

        total_loss += loss.item()

    return total_loss / TRAIN_EPOCHS

def rollout_trajectory(init_state, acs_seq, model):
    """
    init_state: [torch.Tensor, np.ndarray] nx
    acs_seq: [torch.Tensor, np.ndarray] T x nu
    """
    if isinstance(init_state, np.ndarray):
        init_state = torch.from_numpy(init_state)

    if isinstance(acs_seq, np.ndarray):
        acs_seq = torch.from_numpy(acs_seq)

    curr_state = init_state.double().to(TORCH_DEVICE)
    acs_seq = acs_seq.double().to(TORCH_DEVICE)
    for i in range(acs_seq.shape[0]):
        input = torch.cat((curr_state, acs_seq[i]))
        assert len(input.shape) == 1

        state_residual = model(input)
        curr_state += state_residual

    return curr_state


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

        lin0_decays = 0.00025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.0005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.0005 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.00075 * (self.lin3_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data):

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):

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
    ENSEMBLE_SIZE = 5
    nx = 13 # state space size
    nu = 4 # action space size

    model_in = nx + nu # given state and action
    model_out = nx * 2 # predict next state mean and next state variance

    model = PtModel(ENSEMBLE_SIZE, model_in, model_out).to(TORCH_DEVICE)
    model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

    return model