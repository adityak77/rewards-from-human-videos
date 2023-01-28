import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from torch import nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import time
import imageio
import pickle

import argparse
import logging

from sim_env.tabletop import Tabletop
from optimizer_utils import CemLogger, decode_gif, set_all_seeds
from optimizer_utils import load_discriminator_model, load_encoder_model, dvd_reward
from optimizer_utils import tabletop_obs, get_success_values
# from optimizer_utils import vip_reward, vip_reward_trajectory_similarity
from visual_dynamics_model import load_world_model, rollout_trajectory

from distutils.util import strtobool

import sys
sys.path.append('/home/akannan2/rewards-from-human-videos/pydreamer')
from pydreamer import tools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True, help="Task to learn a model for")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed >= 0. If seed < 0, then use random seed")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations of CEM")

    # learned visual dynamics model
    parser.add_argument("--learn_dynamics_model", action='store_true', default=False, help='Learn a dynamics model (otherwise use online sampling)')
    parser.add_argument("--configs", nargs='+', required=True)
    parser.add_argument("--saved_model_path", type=str, default='/home/akannan2/tabletop_dynamics_old/artifacts/checkpoints/latest.pt')

    # optimizer specific
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

    args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = tools.read_yamls('../pydreamer/config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)

    return args, conf


if __name__ == '__main__':
    args, conf = get_args_conf()

    TIMESTEPS = 51 # MPC lookahead - max length of episode
    N_SAMPLES = 100
    NUM_ELITES = 7
    NUM_ITERATIONS = args.num_iter
    nu = 4
    nx = 13

    dtype = torch.double

    # set random seed
    if args.seed >= 0:
        set_all_seeds(args.seed)
        print(f"Using seed: {args.seed}")

    # only one of these reward functions allowed per run
    assert sum([args.dvd, args.vip]) == 1

    if args.learn_dynamics_model:
        assert args.configs and args.saved_model_path

    # assigning reward functions
    if args.dvd:
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

    world_model = load_world_model(conf, args.saved_model_path)
    init_image = imageio.imread('init_state.png')
    init_image = init_image.astype(np.float32) / 255.0 - 0.5

    # reading demos
    if os.path.isdir(args.demo_path):
        all_demo_names = os.listdir(args.demo_path)
        demos = [decode_gif(os.path.join(args.demo_path, fname)) for fname in all_demo_names]
    else:
        demos = [decode_gif(args.demo_path)]

    # initialization
    actions_mean = torch.zeros(TIMESTEPS * nu)
    actions_cov = torch.eye(TIMESTEPS * nu)

    # for logging
    if args.dvd:
        logdir = args.checkpoint.split('/')[1]
    elif args.vip:
        logdir = 'vip'
    
    logdir = 'visual_dynamics_' + logdir + f'_{TIMESTEPS}_{N_SAMPLES}_{NUM_ITERATIONS}_task{args.task_id}'
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

        states = np.zeros((1, TIMESTEPS+1, 120, 180, 3)) # in format [-0.5, 0.5]
        states[0, 0] = init_image
        all_low_dim_states = np.zeros((TIMESTEPS, nx))
        actions = np.zeros((TIMESTEPS, nu))

        # if learned dynamics model then do closed loop
        if args.learn_dynamics_model:
            for t in range(TIMESTEPS):
                print(t)
                tstart = time.time()
                action_distribution = MultivariateNormal(actions_mean, actions_cov)
                action_samples = action_distribution.sample((N_SAMPLES,))
                sample_rewards = torch.zeros(N_SAMPLES)

                ac_seqs = action_samples.reshape(N_SAMPLES, TIMESTEPS, nu)[:, t:]
                assert (t + ac_seqs.shape[1]) == TIMESTEPS
                init_states = np.tile(states[:, t].transpose(0, 3, 1, 2), (N_SAMPLES, 1, 1, 1))
                obs_sampled = rollout_trajectory(init_states, ac_seqs, world_model) # shape B x (T-t) x H x W x C

                prefix_obs = np.tile(states[:, :(t+1)], (N_SAMPLES, 1, 1, 1, 1))
                obs_sampled = np.concatenate((prefix_obs, obs_sampled), axis=1) # shape B x (T+1) x H x W x C

                # revert format of obs_sampled to be 0-1
                obs_sampled += 0.5

                sample_rewards = terminal_reward_fn(obs_sampled, _, demos=demos, video_encoder=video_encoder, sim_discriminator=sim_discriminator)
                print(f'sample_rewards timestep {t}:', sample_rewards.min(), sample_rewards.mean(), sample_rewards.max())

                for i in range(obs_sampled.shape[0]):
                    imageio.mimsave(os.path.join(cem_logger.logdir_iteration, f'dream_traj{i}.gif'), (obs_sampled[i] * 255).astype(np.uint8))

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
                states[0, t+1] = obs
                all_low_dim_states[t] = tabletop_obs(low_dim_info)

                actions[t, :] = action
                tend = time.time()
                print(f'timestep {t} took {tend-tstart} seconds')

                import ipdb; ipdb.set_trace()

        # run open loop CEM if online sampling - not ready yet, and probably not worth
        # integrating with open loop since we can run a separate script to do this.
        else:
            assert False
            for t in range(TIMESTEPS):
                action = traj_sample[0, t*nu:(t+1)*nu].cpu().numpy() # from CEM
                obs, r, done, low_dim_info = env.step(action)
                states[0, t] = obs

        tend = time.perf_counter()
        states[0, 0] += 0.5

        # ALL CODE BELOW for logging sampled trajectory
        additional_reward_type = 'vip' if args.vip else 'dvd'
        additional_reward = terminal_reward_fn(states, _, demos=demos, video_encoder=video_encoder, sim_discriminator=sim_discriminator).item()
        
        # calculate success and reward of trajectory
        any_timestep_succ = False
        for t in range(TIMESTEPS):
            low_dim_state = all_low_dim_states[t]
            _, gt_reward, succ = get_success_values(args.task_id, low_dim_state, very_start)
            
            any_timestep_succ = any_timestep_succ or succ

        low_dim_state = tabletop_obs(low_dim_info)
        rew, gt_reward, succ = get_success_values(args.task_id, low_dim_state, very_start)

        mean_sampled_rewards = sample_rewards.cpu().numpy().mean()

        logger.debug(f"{ep}: Trajectory time: {tend-tstart:.4f}\t Object Position Shift: {rew:.6f}")
        cem_logger.update(gt_reward, succ, any_timestep_succ, mean_sampled_rewards, additional_reward, additional_reward_type)

        # logging results
        all_obs = (states[0] * 255).astype(np.uint8)
        cem_logger.save_graphs(all_obs)

        res_dict = {
            'total_iterations': cem_logger.total_iterations,
            'total_last_success': cem_logger.total_last_success,
            'total_any_timestep_success': cem_logger.total_any_timestep_success,
            'succ_rate_history': cem_logger.succ_rate_history,
            'gt_reward_history': cem_logger.gt_reward_history,
            'seed': args.seed,
        }

        with open(os.path.join(cem_logger.logdir, 'results.pkl'), 'wb') as f:
            pickle.dump(res_dict, f, protocol=pickle.HIGHEST_PROTOCOL)