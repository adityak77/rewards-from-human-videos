import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import time
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
from visual_dynamics_model import load_world_model, rollout_trajectory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('PIL').disabled = True

def run_cem(args, conf):
    TIMESTEPS = 51 # MPC lookahead - max length of episode
    N_SAMPLES = 100
    NUM_ELITES = 7
    NUM_ITERATIONS = args.num_iter
    nu = 4
    nx = 13
    closed_loop_frequency = 10

    # only one of these reward functions allowed per run
    assert sum([args.dvd, args.vip]) == 1
    assert args.configs and args.saved_model_path
    video_encoder, sim_discriminator, terminal_reward_fn = initialize_cem(args)

    # load in models
    world_model = load_world_model(conf, args.saved_model_path)

    # reading demos
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
    cem_logger = initialize_logger(args, TIMESTEPS, N_SAMPLES, NUM_ITERATIONS, prefix='visual_dynamics_')
    for ep in range(NUM_ITERATIONS):
        # env initialization
        env = Tabletop(log_freq=args.env_log_freq, 
                    filepath=args.log_dir + '/env',
                    xml=args.xml,
                    verbose=args.verbose)  # bypass the default TimeLimit wrapper
        _, start_info = env.reset_model()
        very_start = tabletop_obs(start_info)
        init_image = env.get_obs()

        tstart = time.perf_counter()

        states = np.zeros((1, TIMESTEPS+1, 128, 128, 3)) # in format [-0.5, 0.5]
        states[0, 0] = init_image.astype(np.float32) - 0.5
        all_low_dim_states = np.zeros((TIMESTEPS, nx))
        actions = np.zeros((TIMESTEPS, nu))

        for t in range(TIMESTEPS):
            if t % closed_loop_frequency == 0:
                if torch.matrix_rank(actions_cov) < actions_cov.shape[0]:
                    actions_cov += 1e-5 * torch.eye(actions_cov.shape[0])
                action_distribution = MultivariateNormal(actions_mean, actions_cov)
                action_samples = action_distribution.sample((N_SAMPLES,))
                sample_rewards = torch.zeros(N_SAMPLES)

                ac_seqs = action_samples.reshape(N_SAMPLES, TIMESTEPS, nu)[:, t:]
                assert (t + ac_seqs.shape[1]) == TIMESTEPS
                init_states = np.tile(states[:, :(t+1)].transpose(1, 0, 4, 2, 3), (1, N_SAMPLES, 1, 1, 1))
                init_actions = np.tile(np.expand_dims(actions[:(t+1)], axis=1), (1, N_SAMPLES, 1))
                obs_sampled = rollout_trajectory(init_states, init_actions, ac_seqs, world_model) # shape B x (T-t) x H x W x C

                prefix_obs = np.tile(states[:, :(t+1)], (N_SAMPLES, 1, 1, 1, 1))
                obs_sampled = np.concatenate((prefix_obs, obs_sampled), axis=1) # shape B x (T+1) x H x W x C

                # revert format of obs_sampled to be 0-1 for reward calculation
                obs_sampled += 0.5

                if args.dvd:
                    sample_rewards = terminal_reward_fn(obs_sampled, demo_feats=demo_feats, video_encoder=video_encoder, sim_discriminator=sim_discriminator)
                elif args.vip:
                    sample_rewards = terminal_reward_fn(obs_sampled, demos=demos)

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

            action = traj_sample[0, t*nu:(t+1)*nu].cpu().numpy() # from CEM
            obs, r, done, low_dim_info = env.step(action)
            states[0, t+1] = obs - 0.5
            all_low_dim_states[t] = tabletop_obs(low_dim_info)

            actions[t, :] = action

        tend = time.perf_counter()
        ep_time = tend - tstart
        # switch back to [0-1] format
        states[0] += 0.5

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
    args, conf = get_cem_args_conf()
    run_cem(args, conf)
