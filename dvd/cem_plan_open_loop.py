from multi_column import MultiColumn, SimilarityDiscriminator
from utils import remove_module_from_checkpoint_state_dict
from plan_utils import get_args, get_obs
from transforms_video import *

from sim_env.tabletop import Tabletop

import pickle
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
import torchvision
import matplotlib.pyplot as plt
import random
import time
from collections import deque

import importlib
import av
import argparse
import logging
import pickle

from pytorch_mppi.mppi_metaworld import tabletop_obs, evaluate_iteration


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, default=94, help="Task to learn a model for")

parser.add_argument("--log_dir", type=str, default="mppi")
parser.add_argument("--env_log_freq", type=int, default=20) 
parser.add_argument("--verbose", type=bool, default=1) 

parser.add_argument("--xml", type=str, default='env1')

parser.add_argument("--engineered_rewards", action='store_true', default=False, help='Use hand engineered rewards or not')

args = parser.parse_args()

def running_reward_engineered(state, action):
    # task 94: pushing mug right to left
    left_to_right = state[:, 3] - very_start[3]
    # penalty = (left_to_right > 0.1).to(torch.float) * -100
    reward = left_to_right # + penalty

    return reward.to(torch.float32)

if __name__ == '__main__':
    TIMESTEPS = 51 # MPC lookahead
    N_SAMPLES = 100
    NUM_ELITES = 7
    NUM_ITERATIONS = 100
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

    # for logging
    logdir = 'engineered_reward'
    logdir = logdir + f'_{TIMESTEPS}_{N_SAMPLES}_{NUM_ITERATIONS}_open_loop'
    logdir = os.path.join('cem_plots', logdir)
    logdir_iteration = os.path.join(logdir, 'iterations')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    if not os.path.isdir(logdir_iteration):
        os.makedirs(logdir_iteration)

    total_successes = 0
    total_iterations = 0
    rolling_success = deque()
    iteration_reward_history = []
    succ_rate_history = []
    mean_sampled_traj_history = []

    for ep in range(NUM_ITERATIONS):
        tstart = time.perf_counter()
        action_distribution = MultivariateNormal(actions_mean, actions_cov)
        action_samples = action_distribution.sample((N_SAMPLES,))
        sample_rewards = torch.zeros(N_SAMPLES)

        for i in range(N_SAMPLES):
            for t in range(TIMESTEPS):
                u = action_samples[i, nu*t:nu*(t+1)]
                _, _, _, env_info = env.step(u.cpu().numpy())
            
            curr_state = torch.Tensor(tabletop_obs(env_info)).to(device).unsqueeze(0)
            reward = running_reward_engineered(curr_state, u)
            sample_rewards[i] = reward

            env.reset_model()
        
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

        # follow sampled trajectory
        action_distribution = MultivariateNormal(actions_mean, actions_cov)
        traj_sample = action_distribution.sample((1,))
        iters = 0
        done = False
        while not done:
            action = traj_sample[0, iters*nu:(iters+1)*nu] # from CEM
            s, r, done, low_dim_info = env.step(action.cpu().numpy())
            iters += 1
        tend = time.perf_counter()
        rew = tabletop_obs(low_dim_info)[3] - very_start[3]
        logger.debug(f"{ep}: Trajectory time: {tend-tstart:.4f}\t Trajectory reward: {rew:.6f}")

        # calculate success and reward of trajectory
        low_dim_state = tabletop_obs(low_dim_info)
        iteration_reward = low_dim_state[3] - very_start[3]
        # penalty = (iteration_reward > 0.1) * -100
        # iteration_reward += penalty

        succ = iteration_reward > 0.05 # and iteration_reward < 0.1
        
        # print results
        result = 'SUCCESS' if succ else 'FAILURE'
        total_successes += succ
        total_iterations += 1
        rolling_success.append(succ)
        print(f'----------Iteration done: {result} | iteration_reward: {iteration_reward}----------')
        print(f'----------Currently at {total_successes} / {total_iterations}----------')
        
        if len(rolling_success) > 10:
            rolling_success.popleft()
        
        succ_rate = sum(rolling_success) / len(rolling_success)

        iteration_reward_history.append(iteration_reward)
        succ_rate_history.append(succ_rate)
        mean_sampled_traj_history.append(sample_rewards.mean())

        # logging results
        print('----------REPLOTTING----------')
        plt.figure()
        plt.plot([i for i in range(len(iteration_reward_history))], iteration_reward_history)
        plt.xlabel('Iteration')
        plt.ylabel('Total Reward in iteration')
        plt.savefig(os.path.join(logdir, 'engineered_reward_iteration.png'))
        plt.close()
        
        plt.figure()
        plt.plot([i for i in range(len(succ_rate_history))], succ_rate_history)
        plt.xlabel('Iteration')
        plt.ylabel('Rolling Success Rate')
        plt.savefig(os.path.join(logdir, 'rolling_success_rate_iteration.png'))
        plt.close()

        plt.figure()
        plt.plot([i for i in range(len(mean_sampled_traj_history))], mean_sampled_traj_history)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Sampled Trajectories Reward')
        plt.title(f'Mean Reward of Sampled Trajectories')
        plt.savefig(os.path.join(logdir, 'mean_reward_sampled_traj_iteration.png'))
        plt.close()
        
        _, start_info = env.reset_model()
        very_start = tabletop_obs(start_info)