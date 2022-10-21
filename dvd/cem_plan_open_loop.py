from multi_column import MultiColumn, SimilarityDiscriminator
from utils import remove_module_from_checkpoint_state_dict
from transforms_video import *

# from vip import load_vip

from sim_env.tabletop import Tabletop

import pickle
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from torch import nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision
import matplotlib.pyplot as plt
import time
from collections import deque
import imageio
import copy

import importlib
import av
import argparse
import logging
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, required=True, help="Task to learn a model for")

# optimizer specific
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
parser.add_argument('--im_size', type=int, default=120, help='image size to process by encoder')

# env initialization
parser.add_argument("--log_dir", type=str, default="mppi")
parser.add_argument("--env_log_freq", type=int, default=20) 
parser.add_argument("--verbose", type=bool, default=1) 
parser.add_argument("--xml", type=str, default='env1')

args = parser.parse_args()

class CemLogger():
    def __init__(self, logdir):
        print(f'Logging results to {logdir}...')
        self.logdir = logdir
        self.logdir_iteration = os.path.join(logdir, 'iterations')

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.isdir(self.logdir_iteration):
            os.makedirs(self.logdir_iteration)

        self.total_successes = 0
        self.total_iterations = 0
        self.rolling_success = deque()
        self.gt_reward_history = []
        self.succ_rate_history = []
        self.mean_sampled_traj_history = []

    def update(self, gt_reward, succ, mean_sampled_rewards, additional_reward, additional_reward_type):
        # print results
        self.total_successes += succ
        self.total_iterations += 1
        self.rolling_success.append(succ)

        self._display_iteration_result(succ, gt_reward, additional_reward_type, additional_reward)

        # if len(self.rolling_success) > 10:
        #     self.rolling_success.popleft()
        
        succ_rate = sum(self.rolling_success) / len(self.rolling_success)

        self.gt_reward_history.append(gt_reward)
        self.succ_rate_history.append(succ_rate)
        self.mean_sampled_traj_history.append(mean_sampled_rewards)


    def _display_iteration_result(self, gt_reward, succ, additional_reward_type, additional_reward):
        result = 'SUCCESS' if succ else 'FAILURE'
        print(f'----------Iteration done: {result} | gt_reward: {gt_reward:.5f} | {additional_reward_type}_reward {additional_reward:.5f}----------')
        print(f'----------Currently at {self.total_successes} / {self.total_iterations}----------')

    def save_graphs(self, all_obs):
        # Saving graphs of results

        # ground truth rewards
        print('----------REPLOTTING----------')
        plt.figure()
        plt.plot([i for i in range(len(self.gt_reward_history))], self.gt_reward_history)
        plt.xlabel('CEM Iteration')
        plt.ylabel('Total Reward in iteration')
        plt.title('Ground Truth Reward')
        plt.savefig(os.path.join(self.logdir, 'engineered_reward_iteration.png'))
        plt.close()
        
        # rolling average of success rate
        # plt.figure()
        # plt.plot([i for i in range(len(self.succ_rate_history))], self.succ_rate_history)
        # plt.xlabel('CEM Iteration')
        # plt.ylabel('Rolling Success Rate')
        # plt.title('Success Rate (average over last 10 iters)')
        # plt.savefig(os.path.join(self.logdir, 'rolling_success_rate_iteration.png'))
        # plt.close()

        # cumulative success rate
        plt.figure()
        plt.plot([i for i in range(len(self.succ_rate_history))], self.succ_rate_history)
        plt.xlabel('CEM Iteration')
        plt.ylabel('Cumulative Success Rate')
        plt.title('Cumulative Success Rate')
        plt.savefig(os.path.join(self.logdir, 'cumulative_success_rate_iteration.png'))
        plt.close()

        # mean reward of sampled trajectories
        plt.figure()
        plt.plot([i for i in range(len(self.mean_sampled_traj_history))], self.mean_sampled_traj_history)
        plt.xlabel('CEM Iteration')
        plt.ylabel('Mean Sampled Trajectories Reward')
        plt.title('Mean Reward of Sampled Trajectories')
        plt.savefig(os.path.join(self.logdir, 'mean_reward_sampled_traj_iteration.png'))
        plt.close()

        # store video of trajectory
        imageio.mimsave(os.path.join(self.logdir_iteration, f'iteration{ep}.gif'), all_obs, fps=20)

def decode_gif(video_path):
    try: 
        reader = av.open(video_path)
    except:
        print("Issue with opening the video, path:", video_path)
        assert(False)

    return [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]

def load_discriminator_model():
    print("Loading in discriminator model")
    sim_discriminator = SimilarityDiscriminator(args)
    sim_discriminator.load_state_dict(torch.load(args.checkpoint), strict=True)

    return sim_discriminator.to(device)

def load_encoder_model():
    print("Loading in pretrained model")
    cnn_def = importlib.import_module("{}".format('model3D_1'))
    model = MultiColumn(args, args.num_tasks, cnn_def.Model, int(args.hidden_size))
    model_checkpoint = os.path.join('pretrained/video_encoder/', 'model_best.pth.tar')
    model.load_state_dict(remove_module_from_checkpoint_state_dict(torch.load(model_checkpoint)['state_dict']), strict=False)

    return model.to(device)

def inference(states, demo, model, sim_discriminator):
    """
    :param states: (K x T x H x W x C) np.ndarray entries in [0-1]
        T = trajectory size, nx = state embedding size 
    :param demo: (T x H x W x C) List[np.ndarray]

    :returns reward: (K x 1) Tensor
    """
    states = (states * 255).astype(np.uint8)

    transform = ComposeMix([
        [Scale(args.im_size), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(args.im_size), "img"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
         ])

    model.eval()
    sim_discriminator.eval()

    # process states
    def process_batch(samples):
        """
        :param samples: (K x T x H x W x C) np.ndarray
            to prepare before encoding by networks
        """
        K, T, H, W, C = tuple(samples.shape)

        samples = np.swapaxes(samples, 0, 1) # shape T x K x H x W x C
        sample_downsample = samples[1::max(1, len(samples) // 30)][:30] # downsample 30 x K x H x W x C
        sample_downsample = np.swapaxes(sample_downsample, 0, 1) # shape K x 30 x H x W x C

        sample_flattened = np.reshape(sample_downsample, (-1, H, W, C)) # Flatten as (K x 30) x H x W x C
        sample_flattened = [elem for elem in sample_flattened] # need to convert to list to make PIL image conversion

        # sample_transform should be K x 30 x C x 120 x 120 after cropping and trajectory downsampling
        sample_transform = torch.stack(transform(sample_flattened))
        sample_transform = sample_transform.reshape(K, 30, C, 120, 120).permute(0, 2, 1, 3, 4)
        sample_data = [sample_transform.to(device)]

        return sample_data
    
    demo_data = process_batch(np.expand_dims(np.array(demo), axis=0))
    states_data = process_batch(states)

    # evaluate trajectory reward
    with torch.no_grad():
        states_enc = model.encode(states_data) # K x 512
        demo_enc = model.encode(demo_data) # 1 x 512

        logits = sim_discriminator.forward(states_enc, demo_enc.repeat(states_enc.shape[0], 1))
        reward_samples = F.softmax(logits, dim=1)

    return reward_samples[:, 1]

def dvd_reward(states, actions):
    """
    :param states: (K x T x H x W x C) np.ndarray entries in [0-1]
        K = batch size, T = trajectory size, nx = state embedding size

    :param actions: (K x T x nu) Tensor

    :returns rewards: (K x 1) Tensor
    """
    CHUNKS = 100
    command_start = time.perf_counter()
    rewards_demo = torch.zeros(states.shape[0])
    for demo in demos:
        reward_list = []
        batch_num = 0
        while batch_num * CHUNKS < states.shape[0]:
            loc = slice(batch_num*CHUNKS, max((batch_num+1)*CHUNKS, states.shape[0]))
            rewards_batch = inference(states[loc], demo, video_encoder, sim_discriminator)
            reward_list.append(rewards_batch.cpu())
            torch.cuda.empty_cache()
            batch_num += 1

        rewards_demo += torch.cat(reward_list, dim=0)
    rewards = rewards_demo / len(demos)
    elapsed = time.perf_counter() - command_start
    print(f"Video similarity inference elapsed time: {elapsed:.4f}s")

    return rewards

def vip_reward(states, actions):
    """
    :param states: (K x T x H x W x C) np.ndarray entries in [0-1]
        K = batch size, T = trajectory size, nx = state embedding size

    :param actions: (K x T x nu) Tensor

    :returns rewards: (K x 1) Tensor
    """
    vip = load_vip().to(device)
    vip.eval()

    def preprocess(input):
        transforms = ComposeMix([
            [torchvision.transforms.ToPILImage(), "img"],
            [torchvision.transforms.Resize(256), "img"],
            [torchvision.transforms.CenterCrop(224), "img"],
            [torchvision.transforms.ToTensor(), "img"],
         ])

        output = [(elem * 255).astype(np.uint8) for elem in input]
        output = torch.stack(transforms(output)) # K x C x 224 x 224
        output = (output * 255).to(torch.uint8).to(device)
        return output

    final_states = preprocess(states[:, -1, :, :, :])
    demo_goals = preprocess([demo[-1] for demo in demos])

    with torch.no_grad():
        final_states_emb = vip(final_states)
        demo_goals_emb = vip(demo_goals)

    reward = torch.zeros(final_states_emb.shape[0])
    for demo in demo_goals_emb:
        reward += torch.linalg.norm(final_states_emb - demo, dim=1).cpu()

    reward /= demo_goals_emb.shape[0]
    return -reward

def vip_reward_trajectory_similarity(states, actions):
    """
    :param states: (K x T x H x W x C) np.ndarray entries in [0-1]
        K = batch size, T = trajectory size, nx = state embedding size

    :param actions: (K x T x nu) Tensor

    :returns rewards: (K x 1) Tensor
    """
    vip = load_vip().to(device)
    vip.eval()

    def preprocess(input):
        transforms = ComposeMix([
            [torchvision.transforms.ToPILImage(), "img"],
            [torchvision.transforms.Resize(256), "img"],
            [torchvision.transforms.CenterCrop(224), "img"],
            [torchvision.transforms.ToTensor(), "img"],
         ])

        output = [(elem * 255).astype(np.uint8) for elem in input]
        output = torch.stack(transforms(output)) # K x C x 224 x 224
        output = (output * 255).to(torch.uint8).to(device)
        return output

    initial_states = preprocess(states[:, 0, :, :, :])    
    final_states = preprocess(states[:, -1, :, :, :])
    demo_starts = preprocess([demo[0] for demo in demos])
    demo_goals = preprocess([demo[-1] for demo in demos])

    with torch.no_grad():
        initial_states_emb = vip(initial_states)
        final_states_emb = vip(final_states)
        demo_starts_emb = vip(demo_starts)
        demo_goals_emb = vip(demo_goals)

    traj_emb = final_states_emb - initial_states_emb
    demo_traj_emb = demo_goals_emb - demo_starts_emb

    reward = torch.zeros(final_states_emb.shape[0])
    for demo in demo_traj_emb:
        reward += torch.linalg.norm(demo - traj_emb, dim=1).cpu()

    reward /= demo_goals_emb.shape[0]
    return -reward

def reward_push_mug_left_to_right(state, action):
    ''' 
    task 94: pushing mug right to left
    :param state: (nx) np.ndarray
        output from tabletop_obs
    :param action: (T x nu) Tensor

    :returns scalar Tensor
    '''
    left_to_right = -np.abs(state[3] - very_start[3] - 0.15) + 0.15
    drawer_move = np.abs(state[10] - very_start[10]) 
    move_faucet = state[12]

    penalty = 0 if (drawer_move < 0.03 and move_faucet < 0.01) else -100
    reward = left_to_right + penalty

    return torch.Tensor([reward]).to(torch.float32)

def reward_push_mug_forward(state, action):
    '''
    task 41: pushing mug away from camera
    :param state: (nx) np.ndarray
        output from tabletop_obs
    :param action: (T x nu) Tensor

    :returns scalar Tensor
    '''
    x_shift = np.abs(state[3] - very_start[3])
    forward = -np.abs(state[4] - very_start[4] - 0.115) + 0.115

    penalty = 0 if (x_shift < 0.05) else -100
    reward = forward + penalty

    return torch.Tensor([reward]).to(torch.float32)

def reward_close_drawer(state, action):
    '''
    task 5: closing drawer
    :param state: (K x nx) np.ndarray
        output from tabletop_obs
    :param action: (K x T x nu) Tensor

    :returns scalar Tensor
    '''
    right_to_left = np.abs(state[3] - very_start[3])
    closed = state[10]

    penalty = 0 if right_to_left < 0.01 else -100
    reward = closed + penalty

    return torch.Tensor([reward]).to(torch.float32)

def tabletop_obs(info):
    '''
    Convert env_info outputs from env.step() function into state
    '''
    hand = np.array([info['hand_x'], info['hand_y'], info['hand_z']])
    mug = np.array([info['mug_x'], info['mug_y'], info['mug_z']])
    mug_quat = np.array([info['mug_quat_x'], info['mug_quat_y'], info['mug_quat_z'], info['mug_quat_w']])
    init_low_dim = np.concatenate([hand, mug, mug_quat, [info['drawer']], [info['coffee_machine']], [info['faucet']]])
    return init_low_dim

if __name__ == '__main__':
    TIMESTEPS = 51 # MPC lookahead - max length of episode
    N_SAMPLES = 100
    NUM_ELITES = 7
    NUM_ITERATIONS = 100
    nu = 4

    dtype = torch.double

    # only one of these reward functions allowed per run
    assert sum([args.engineered_rewards, args.dvd, args.vip]) == 1

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
        video_encoder = load_encoder_model()
        sim_discriminator = load_discriminator_model()
        terminal_reward_fn = dvd_reward
    elif args.vip:
        assert args.demo_path is not None
        if args.demo_path.startswith('demos'):
            assert args.demo_path.endswith(str(args.task_id))
        terminal_reward_fn = vip_reward

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
            states = np.zeros((N_SAMPLES, TIMESTEPS, 120, 180, 3))

        for i in range(N_SAMPLES):
            env_copy = copy.deepcopy(env)
            for t in range(TIMESTEPS):
                u = action_samples[i, t*nu:(t+1)*nu]
                obs, _, _, env_copy_info = env_copy.step(u.cpu().numpy())
                if args.dvd or args.vip:
                    states[i, t] = obs
            
            if args.engineered_rewards:
                curr_state = tabletop_obs(env_copy_info)
                sample_rewards[i] = terminal_reward_fn(curr_state, u)

        if args.dvd or args.vip:
            sample_rewards = terminal_reward_fn(states, _)

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

        states = np.zeros((1, TIMESTEPS, 120, 180, 3))

        for t in range(TIMESTEPS):
            action = traj_sample[0, t*nu:(t+1)*nu] # from CEM
            obs, r, done, low_dim_info = env.step(action.cpu().numpy())
            states[0, t] = obs
        tend = time.perf_counter()

        # ALL CODE BELOW for logging sampled trajectory
        additional_reward_type = 'vip' if args.vip else 'dvd'
        if not args.engineered_rewards:
            additional_reward = terminal_reward_fn(states, _).item()
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
        all_obs = (states[0] * 255).astype(np.uint8)
        cem_logger.save_graphs(all_obs)