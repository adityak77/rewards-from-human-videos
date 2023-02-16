import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from torch import nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import time
import copy
import logging

from sim_env.tabletop import Tabletop

from optimizer_utils import (get_cem_args_conf, 
                             initialize_cem, 
                             initialize_logger, 
                             cem_iteration_logging, 
                             decode_gif, 
                             dvd_process_encode_batch, 
                             tabletop_obs
                            )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True

def run_cem(args):
    TIMESTEPS = 51 # MPC lookahead - max length of episode
    N_SAMPLES = 100
    NUM_ELITES = 7
    NUM_ITERATIONS = args.num_iter
    nu = 4

    # only one of these reward functions allowed per run
    video_encoder, sim_discriminator, terminal_reward_fn = initialize_cem(args)

    if not args.engineered_rewards:
        if os.path.isdir(args.demo_path):
            all_demo_names = os.listdir(args.demo_path)
            demos = [decode_gif(os.path.join(args.demo_path, fname)) for fname in all_demo_names]
        else:
            demos = [decode_gif(args.demo_path)]

        if args.dvd:
            demo_feats = [dvd_process_encode_batch(np.array([demo]), video_encoder) for demo in demos]

    # initialization
    actions_mean = torch.zeros(TIMESTEPS * nu)
    actions_cov = torch.eye(TIMESTEPS * nu)

    # for logging
    cem_logger = initialize_logger(args, TIMESTEPS, N_SAMPLES, NUM_ITERATIONS)
    for ep in range(NUM_ITERATIONS):
        # env initialization
        env = Tabletop(log_freq=args.env_log_freq, 
                    filepath=args.log_dir + '/env',
                    xml=args.xml,
                    verbose=args.verbose)  # bypass the default TimeLimit wrapper
        _, start_info = env.reset_model()
        very_start = tabletop_obs(start_info)

        tstart = time.perf_counter()
        if torch.matrix_rank(actions_cov) < actions_cov.shape[0]:
            actions_cov += 1e-5 * torch.eye(actions_cov.shape[0])
        action_distribution = MultivariateNormal(actions_mean, actions_cov)
        action_samples = action_distribution.sample((N_SAMPLES,))
        sample_rewards = torch.zeros(N_SAMPLES)

        if args.dvd or args.vip:
            states = np.zeros((N_SAMPLES, TIMESTEPS, 128, 128, 3))

        for i in range(N_SAMPLES):
            env_copy = copy.deepcopy(env)
            for t in range(TIMESTEPS):
                u = action_samples[i, t*nu:(t+1)*nu]
                obs, _, _, env_copy_info = env_copy.step(u.cpu().numpy())
                if args.dvd or args.vip:
                    states[i, t] = obs
            
            if args.engineered_rewards:
                curr_state = tabletop_obs(env_copy_info)
                sample_rewards[i] = terminal_reward_fn(curr_state, u, very_start=very_start)

        if args.dvd:
            sample_rewards = terminal_reward_fn(states, demo_feats=demo_feats, video_encoder=video_encoder, sim_discriminator=sim_discriminator)
        elif args.vip:
            sample_rewards = terminal_reward_fn(states, demos=demos)

        # update elites
        _, best_inds = torch.topk(sample_rewards, NUM_ELITES)
        elites = action_samples[best_inds]

        actions_mean = elites.mean(dim=0)
        actions_cov = torch.Tensor(np.cov(elites.cpu().numpy().T))
        if torch.matrix_rank(actions_cov) < actions_cov.shape[0]:
            actions_cov += 1e-5 * torch.eye(actions_cov.shape[0])

        # follow sampled trajectory
        action_distribution = MultivariateNormal(actions_mean, actions_cov)
        traj_sample = action_distribution.sample((1,))

        states = np.zeros((1, TIMESTEPS, 128, 128, 3))
        all_low_dim_states = np.zeros((TIMESTEPS, 13))

        for t in range(TIMESTEPS):
            action = traj_sample[0, t*nu:(t+1)*nu] # from CEM
            obs, r, done, low_dim_info = env.step(action.cpu().numpy())
            states[0, t] = obs
            all_low_dim_states[t] = tabletop_obs(low_dim_info)
        tend = time.perf_counter()
        ep_time = tend - tstart

        # logging sampled trajectory
        if args.dvd:
            cem_iteration_logging(args, TIMESTEPS, ep, ep_time, cem_logger, states, all_low_dim_states, very_start, 
                                  sample_rewards, calculate_add_reward=True, terminal_reward_fn=terminal_reward_fn, 
                                  video_encoder=video_encoder, sim_discriminator=sim_discriminator, 
                                  demo_feats=demo_feats)
        elif args.vip:
            cem_iteration_logging(args, TIMESTEPS, ep, ep_time, cem_logger, states, all_low_dim_states, very_start, 
                                  sample_rewards, calculate_add_reward=True, terminal_reward_fn=terminal_reward_fn, 
                                  demos=demos)

if __name__ == '__main__':
    args, _ = get_cem_args_conf()
    run_cem(args)
