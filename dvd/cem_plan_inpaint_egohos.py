import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from torch import nn as nn
import torch
import torch.multiprocessing as mp
from torch.distributions.multivariate_normal import MultivariateNormal
import time
import functools
import copy
import cv2
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

from inpaint_utils import get_robot_cfg, get_inpaint_model, get_segmentation_model_egohos, inpaint, inpaint_wrapper, inpaint_egohos

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

    # only dvd for inpainting
    assert args.dvd
    video_encoder, sim_discriminator, terminal_reward_fn = initialize_cem(args)

    # load in models
    robot_cfg = get_robot_cfg()
    human_segmentation_model = get_segmentation_model_egohos()
    inpaint_model = get_inpaint_model(args)

    if os.path.isdir(args.demo_path):
        all_demo_names = os.listdir(args.demo_path)
        demos = [decode_gif(os.path.join(args.demo_path, fname)) for fname in all_demo_names]
    else:
        demos = [decode_gif(args.demo_path)]

    # Inpaint demos here (and maintain same formatting)
    demos = [inpaint_egohos(args, inpaint_model, human_segmentation_model, demo) for demo in demos]
    demo_feats = [dvd_process_encode_batch(np.array([demo]), video_encoder) for demo in demos]

    # initialization
    actions_mean = torch.zeros(TIMESTEPS * nu)
    actions_cov = torch.eye(TIMESTEPS * nu)

    # for logging
    cem_logger = initialize_logger(args, TIMESTEPS, N_SAMPLES, NUM_ITERATIONS, prefix='inpaint_egohos_')
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

        if args.dvd or args.vip:
            # Inpaint states here
            states = (states * 255).astype(np.uint8)

            # reshape trajectories to have 30 timesteps instead of 51
            downsample = TIMESTEPS // 30 + 1
            if downsample > 1:
                downsample_boundary = (TIMESTEPS - 30) // (downsample - 1) * downsample
                states = np.concatenate((states[:, :downsample_boundary:downsample, :, :, :], states[:, downsample_boundary:, :, :, :]), axis=1)
                states = states[:, :30, :, :, :]

            if not args.no_robot_inpaint:
                # detectron2 input is BGR
                for i in range(len(states)):
                    for j in range(len(states[i])):
                        states[i][j] = cv2.cvtColor(states[i][j], cv2.COLOR_RGB2BGR)

                inpaint_wrapper_parallel = functools.partial(inpaint_wrapper, args, states, robot_cfg)
                with mp.Pool(processes=args.num_gpus) as p:
                    inpainted_states = p.map(inpaint_wrapper_parallel, list(range(args.num_gpus)))
                    states = np.concatenate(inpainted_states, axis=0)

                # convert back to RGB
                for i in range(len(states)):
                    for j in range(len(states[i])):
                        states[i][j] = cv2.cvtColor(states[i][j], cv2.COLOR_BGR2RGB)
    
            if args.dvd:
                sample_rewards = terminal_reward_fn(states, _, demo_feats=demo_feats, video_encoder=video_encoder, sim_discriminator=sim_discriminator)
            elif args.vip:
                sample_rewards = terminal_reward_fn(states, _, demos=demos)

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
        cem_iteration_logging(args, TIMESTEPS, ep, ep_time, cem_logger, states, all_low_dim_states, very_start, sample_rewards)

def main():
    mp.set_start_method('spawn')
    args, _ = get_cem_args_conf()
    args.num_gpus = len(list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
    run_cem(args)

if __name__ == '__main__':
    main()