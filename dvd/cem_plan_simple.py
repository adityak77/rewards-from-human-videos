from sim_env.tabletop import Tabletop

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import torch.optim as optim
from torch import nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import argparse
import logging

from pytorch_mppi.mppi_metaworld import tabletop_obs


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, default=94, help="Task to learn a model for")
parser.add_argument("--xml", type=str, default='env1')
parser.add_argument("--engineered_rewards", action='store_true', default=False, help='Use hand engineered rewards or not')

args = parser.parse_args()

def running_reward_engineered(state, action):
    # task 94: pushing mug right to left
    left_to_right = state[:, 3]

    return left_to_right.to(torch.float32)

if __name__ == '__main__':
    TIMESTEPS = 51 # MPC lookahead
    N_SAMPLES = 100
    NUM_ELITES = 7
    NUM_EPISODES = 100
    nu = 4

    dtype = torch.double

    # env initialization
    env = Tabletop(log_freq=args.env_log_freq, 
                   filepath=args.log_dir + '/env',
                   xml=args.xml,
                   verbose=args.verbose)  # bypass the default TimeLimit wrapper
    _, start_info = env.reset_model()
    very_start = tabletop_obs(start_info)

    # initialization
    actions_mean = torch.zeros(TIMESTEPS * nu)
    actions_cov = torch.eye(TIMESTEPS * nu)

    for ep in range(NUM_EPISODES):
        done = False
        iters = 0
        step_rewards = []
        while not done:
            action_distribution = MultivariateNormal(actions_mean, actions_cov)
            action_samples = action_distribution.sample((N_SAMPLES,))
            sample_rewards = torch.zeros(N_SAMPLES)

            saved_env_state = env.get_env_state()
            saved_path_length = env.cur_path_length
            for i in range(N_SAMPLES):
                for t in range(iters, TIMESTEPS):
                    u = action_samples[i, nu*t:nu*(t+1)]
                    _, _, _, env_info = env.step(u.cpu().numpy())
                    curr_state = torch.Tensor(tabletop_obs(env_info)).to(device).unsqueeze(0)
                    reward = running_reward_engineered(curr_state, u)
                    sample_rewards[i] = sample_rewards[i] + reward

                env.set_env_state(saved_env_state)
                env.cur_path_length = saved_path_length
            
            # update elites
            _, best_inds = torch.topk(sample_rewards, NUM_ELITES)
            elites = action_samples[best_inds]

            actions_mean = elites.mean(dim=0)
            actions_cov = torch.Tensor(np.cov(elites.cpu().numpy().T))
            for i in range(actions_cov.shape[0]):
                for j in range(actions_cov.shape[1]):
                    if i != j:
                        actions_cov[i, j] = 0
                    else:
                        actions_cov[i, j] = max(actions_cov[i,j], 1e-8)

            # MPC step
            action = elites[0, iters*nu:(iters+1)*nu] # from CEM
            s, r, done, low_dim_info = env.step(action.cpu().numpy())
            iters += 1
            rew = tabletop_obs(low_dim_info)[3] - very_start[3]

        # calculate success and reward of trajectory
        low_dim_state = tabletop_obs(low_dim_info)
        episode_reward = low_dim_state[3] - very_start[3]
        succ = episode_reward > 0.05
        
        # print results
        _, start_info = env.reset_model()
        very_start = tabletop_obs(start_info)