import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from torch import nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import time
import copy

import argparse
import logging

from sim_env.tabletop import Tabletop
from optimizer_utils import CemLogger, decode_gif
from optimizer_utils import load_discriminator_model, load_encoder_model, dvd_reward
from optimizer_utils import reward_push_mug_left_to_right, reward_push_mug_forward, reward_close_drawer, tabletop_obs
# from optimizer_utils import vip_reward, vip_reward_trajectory_similarity
from state_dynamics_model import Dataset, nn_constructor, rollout_trajectory, train


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True, help="Task to learn a model for")

# optimizer specific
parser.add_argument("--learn_dynamics_model", action='store_true', default=False, help='Learn a dynamics model (otherwise use online sampling)')
parser.add_argument("--engineered_rewards", action='store_true', default=False, help='Use hand engineered rewards or not')
parser.add_argument("--dvd", action='store_true', help='Use dvd rewards')
parser.add_argument("--vip", action='store_true', help='Use pretrained VIP embeddings for reward function')

# for DVD or VIP rewards
parser.add_argument("--demo_path", type=str, default=None, help='path to demo video')

# dvd model params
parser.add_argument("--checkpoint", type=str, default='test/tasks6_seed0_lr0.01_sim_pre_hum54144469394_dem60_rob54193/model/150sim_discriminator.pth.tar', help='path to model')
parser.add_argument('--similarity', action='store_true', default=True, help='whether to use similarity discriminator') # needs to be true for MultiColumn init
parser.add_argument('--hidden_size', type=int, default=512, help='latent encoding size')
parser.add_argument('--num_tasks', type=int, default=2, help='number of tasks') # needs to exist for MultiColumn init

# env initialization
parser.add_argument("--log_dir", type=str, default="mppi")
parser.add_argument("--env_log_freq", type=int, default=20) 
parser.add_argument("--verbose", type=bool, default=1) 
parser.add_argument("--xml", type=str, default='env1')

args = parser.parse_args()


if __name__ == '__main__':
    TIMESTEPS = 51 # MPC lookahead - max length of episode
    N_SAMPLES = 100
    NUM_ELITES = 7
    NUM_ITERATIONS = 100
    nu = 4
    nx = 13

    dtype = torch.double

    # only one of these reward functions allowed per run
    assert sum([args.engineered_rewards, args.dvd, args.vip]) == 1

    # assigning reward functions
    if args.engineered_rewards:
        if args.task_id == 94:
            terminal_reward_fn = reward_push_mug_left_to_right
        elif args.task_id == 41:
            terminal_reward_fn = reward_push_mug_forward
        elif args.task_id == 5:
            terminal_reward_fn = reward_close_drawer
    elif args.dvd:
        assert args.demo_path is not None
        if args.demo_path.startswith('demos'):
            assert args.demo_path.endswith(str(args.task_id))
        video_encoder = load_encoder_model(args)
        sim_discriminator = load_discriminator_model(args)
        terminal_reward_fn = dvd_reward
    elif args.vip:
        assert args.demo_path is not None
        if args.demo_path.startswith('demos'):
            assert args.demo_path.endswith(str(args.task_id))
        video_encoder = None
        sim_discriminator = None
        terminal_reward_fn = vip_reward

    # dynamics function loading
    if args.engineered_rewards and args.learn_dynamics_model:
        dataset = Dataset(nx, nu)
        model = nn_constructor()

    # reading demos
    if not args.engineered_rewards:
        if os.path.isdir(args.demo_path):
            all_demo_names = os.listdir(args.demo_path)
            demos = [decode_gif(os.path.join(args.demo_path, fname)) for fname in all_demo_names]
        else:
            demos = [decode_gif(args.demo_path)]

    # initialization
    actions_mean = torch.zeros(TIMESTEPS * nu)
    actions_cov = torch.eye(TIMESTEPS * nu)

    # for logging
    if args.engineered_rewards:
        logdir = 'engineered_reward'
    elif args.dvd:
        logdir = args.checkpoint.split('/')[1]
    elif args.vip:
        logdir = 'vip2'
    
    logdir = logdir + f'_{TIMESTEPS}_{N_SAMPLES}_{NUM_ITERATIONS}_task{args.task_id}'
    logdir = os.path.join('cem_plots', logdir)
    run = 0
    while os.path.isdir(logdir + f'_run{run}'):
        run += 1
    logdir = logdir + f'_run{run}'
    
    cem_logger = CemLogger(logdir)
    for ep in range(NUM_ITERATIONS):
        # env initialization
        env = Tabletop(log_freq=args.env_log_freq, 
                    filepath=args.log_dir + '/env',
                    xml=args.xml,
                    verbose=args.verbose)  # bypass the default TimeLimit wrapper
        _, start_info = env.reset_model()
        very_start = tabletop_obs(start_info)

        tstart = time.perf_counter()
        action_distribution = MultivariateNormal(actions_mean, actions_cov)
        action_samples = action_distribution.sample((N_SAMPLES,))
        sample_rewards = torch.zeros(N_SAMPLES)

        if args.dvd or args.vip:
            all_obs = np.zeros((N_SAMPLES, TIMESTEPS, 120, 180, 3))

        for i in range(N_SAMPLES):
            """Abstract away below"""
            env_copy = copy.deepcopy(env)
            if args.engineered_rewards and args.learn_dynamics_model:
                ac_seqs = action_samples[i].reshape(TIMESTEPS, nu)
                final_state = rollout_trajectory(very_start, ac_seqs, model)
            else:
                for t in range(TIMESTEPS):
                    u = action_samples[i, t*nu:(t+1)*nu]
                    obs, _, _, env_copy_info = env_copy.step(u.cpu().numpy())
                    if args.dvd or args.vip:
                        all_obs[i, t] = obs
                final_state = tabletop_obs(env_copy_info)
            """"Abstract away above"""

            if args.engineered_rewards:
                sample_rewards[i] = terminal_reward_fn(final_state, _, very_start=very_start)

        if args.dvd or args.vip:
            sample_rewards = terminal_reward_fn(all_obs, _, demos=demos, video_encoder=video_encoder, sim_discriminator=sim_discriminator)

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

        all_obs = np.zeros((1, TIMESTEPS, 120, 180, 3))

        states = np.zeros((TIMESTEPS+1, nx))
        actions = np.zeros((TIMESTEPS, nu))
        states[0] = very_start
        for t in range(TIMESTEPS):
            action = traj_sample[0, t*nu:(t+1)*nu] # from CEM
            obs, r, done, low_dim_info = env.step(action.cpu().numpy())
            all_obs[0, t] = obs

            states[t+1, :] = tabletop_obs(low_dim_info)
            actions[t, :] = action.cpu().numpy()

        # add data to dataset and training model
        dataset.add(states, actions)
        train(model, dataset)

        tend = time.perf_counter()

        # ALL CODE BELOW for logging sampled trajectory
        additional_reward_type = 'vip' if args.vip else 'dvd'
        if not args.engineered_rewards:
            additional_reward = terminal_reward_fn(all_obs, _, demos=demos, video_encoder=video_encoder, sim_discriminator=sim_discriminator).item()
        else:
            additional_reward = 0 # NA

        low_dim_state = tabletop_obs(low_dim_info)
        # TODO: fix reward section below / integrate with gt rewards above
        if args.task_id == 94:
            rew = low_dim_state[3] - very_start[3]
            gt_reward = -np.abs(low_dim_state[3] - very_start[3] - 0.15) + 0.15
            penalty = 0 if (np.abs(low_dim_state[10] - very_start[10]) < 0.03 and low_dim_state[12] < 0.01) else -100
            success_threshold = 0.05
        elif args.task_id == 41:
            rew = low_dim_state[4] - very_start[4]
            gt_reward = -np.abs(low_dim_state[4] - very_start[4] - 0.115) + 0.115
            penalty = 0  # if (np.abs(low_dim_state[3] - very_start[3]) < 0.05) else -100
            success_threshold = 0.03
        elif args.task_id == 5:
            rew = low_dim_state[10]
            gt_reward = low_dim_state[10]
            penalty = 0 if (np.abs(low_dim_state[3] - very_start[3]) < 0.01) else -100
            success_threshold = -0.01

        # calculate success and reward of trajectory
        gt_reward += penalty
        succ = gt_reward > success_threshold
        mean_sampled_rewards = sample_rewards.cpu().numpy().mean()

        logger.debug(f"{ep}: Trajectory time: {tend-tstart:.4f}\t Object Position Shift: {rew:.6f}")
        cem_logger.update(gt_reward, succ, mean_sampled_rewards, additional_reward, additional_reward_type)

        # logging results
        all_obs = (all_obs[0] * 255).astype(np.uint8)
        cem_logger.save_graphs(all_obs)