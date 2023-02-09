import os
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import math
from PIL import Image
import time
from collections import deque
import warnings
import random

# from vip import load_vip

import av
import importlib
from multi_column import MultiColumn, SimilarityDiscriminator
from utils import remove_module_from_checkpoint_state_dict
from transforms_video import *

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CemLogger():
    def __init__(self, logdir):
        print(f'Logging results to {logdir}...')
        self.logdir = logdir
        self.logdir_iteration = os.path.join(logdir, 'iterations')

        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.isdir(self.logdir_iteration):
            os.makedirs(self.logdir_iteration)

        self.total_iterations = 0
        self.total_last_success = 0
        self.rolling_last_success = deque()
        self.total_any_timestep_success = 0
        self.rolling_any_timestep_success = deque()
        self.gt_reward_history = []
        self.succ_rate_history = []
        self.mean_sampled_traj_history = []

    def update(self, gt_reward, last_succ, any_timestep_succ, mean_sampled_rewards, additional_reward, additional_reward_type):
        # print results
        self.total_iterations += 1
        self.total_last_success += last_succ
        self.rolling_last_success.append(last_succ)
        self.total_any_timestep_success += any_timestep_succ
        self.rolling_any_timestep_success.append(any_timestep_succ)

        self._display_iteration_result(gt_reward, last_succ, any_timestep_succ, additional_reward_type, additional_reward)

        # if len(self.rolling_last_success) > 10:
        #     self.rolling_last_success.popleft()
        
        succ_rate = sum(self.rolling_any_timestep_success) / len(self.rolling_any_timestep_success)

        self.gt_reward_history.append(gt_reward)
        self.succ_rate_history.append(succ_rate)
        self.mean_sampled_traj_history.append(mean_sampled_rewards)


    def _display_iteration_result(self, gt_reward, last_succ, any_timestep_succ, additional_reward_type, additional_reward):
        result = 'SUCCESS' if last_succ else 'FAILURE'
        any_timestep_result = 'SUCCESS' if any_timestep_succ else 'FAILURE'
        print(f'----------Iteration done: {result} / {any_timestep_result} (any_timestep) | gt_reward: {gt_reward:.5f} | {additional_reward_type}_reward {additional_reward:.5f}----------')
        print(f'----------Currently at {self.total_last_success} / {self.total_iterations}----------')
        print(f'----------Currently at {self.total_any_timestep_success} / {self.total_iterations}----------')

    def save_graphs(self, all_obs, all_obs_inpainted=None):
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
        imageio.mimsave(os.path.join(self.logdir_iteration, f'iteration{self.total_iterations}.gif'), all_obs, fps=20)
        if all_obs_inpainted is not None:
            imageio.mimsave(os.path.join(self.logdir_iteration, f'iteration{self.total_iterations}_inpainted.gif'), all_obs_inpainted, fps=20)

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def decode_gif(video_path):
    try: 
        reader = av.open(video_path)
    except:
        print("Issue with opening the video, path:", video_path)
        assert(False)

    return [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]

def load_discriminator_model(args):
    print("Loading in discriminator model")
    sim_discriminator = SimilarityDiscriminator(args)
    sim_discriminator.load_state_dict(torch.load(args.checkpoint), strict=True)

    return sim_discriminator.to(device)

def load_encoder_model(args):
    print("Loading in pretrained model")
    cnn_def = importlib.import_module("{}".format('model3D_1'))
    model = MultiColumn(args, args.num_tasks, cnn_def.Model, int(args.hidden_size))
    model_checkpoint = os.path.join('pretrained/video_encoder/', 'model_best.pth.tar')
    model.load_state_dict(remove_module_from_checkpoint_state_dict(torch.load(model_checkpoint)['state_dict']), strict=False)

    return model.to(device)

def dvd_reward(states, actions, **kwargs):
    """
    :param states: (K x T x H x W x C) np.ndarray entries in [0-1]
        K = batch size, T = trajectory size, nx = state embedding size

    :param actions: (K x T x nu) Tensor

    :returns rewards: (K x 1) Tensor
    """
    demo_feats = kwargs['demo_feats']
    video_encoder = kwargs['video_encoder']
    sim_discriminator = kwargs['sim_discriminator']
    
    rewards_demo = torch.zeros(states.shape[0], device=device)
    states = (states * 255).astype(np.uint8)
    states_feats = dvd_process_encode_batch(states, video_encoder)
    
    sim_discriminator.eval()
    with torch.no_grad():
        for demo_feat in demo_feats:
            rewards_list = []
            for state_feat in states_feats:
                logits = sim_discriminator.forward(state_feat.unsqueeze(0), demo_feat)
                reward_samples = F.softmax(logits, dim=1)
                rewards_list.append(reward_samples[:, 1])

            rewards_demo += torch.cat(rewards_list, dim=0)
    rewards = rewards_demo / len(demo_feats)

    return rewards

def dvd_process_encode_batch(samples, video_encoder):
    """
    :param samples: (K x T x H x W x C) np.ndarray
        to prepare before encoding by networks
    """
    transform = ComposeMix([
        [Scale(120), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(120), "img"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
         ])

    K, T, H, W, C = tuple(samples.shape)

    samples = np.swapaxes(samples, 0, 1) # shape T x K x H x W x C
    sample_downsample = samples[::max(1, len(samples) // 30)][:30] # downsample 30 x K x H x W x C
    sample_downsample = np.swapaxes(sample_downsample, 0, 1) # shape K x 30 x H x W x C

    sample_flattened = np.reshape(sample_downsample, (-1, H, W, C)) # Flatten as (K x 30) x H x W x C
    sample_flattened = [elem for elem in sample_flattened] # need to convert to list to make PIL image conversion

    # sample_transform should be K x 30 x C x 120 x 120 after cropping and trajectory downsampling
    sample_transform = torch.stack(transform(sample_flattened))
    sample_transform = sample_transform.reshape(K, 30, C, 120, 120).permute(0, 2, 1, 3, 4)
    sample_data = [sample_transform.to(device)]

    video_encoder.eval()
    with torch.no_grad():
        states_enc = video_encoder.encode(sample_data) # K x 512

    return states_enc

def vip_reward(states, actions, **kwargs):
    """
    :param states: (K x T x H x W x C) np.ndarray entries in [0-1]
        K = batch size, T = trajectory size, nx = state embedding size

    :param actions: (K x T x nu) Tensor

    :returns rewards: (K x 1) Tensor
    """
    demos = kwargs['demos']

    vip = load_vip().to(device)
    vip.eval()

    def preprocess(input):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
         ])

        output = [Image.fromarray((elem * 255).astype(np.uint8)) for elem in input]
        output = torch.stack([transforms(elem) for elem in output]) # K x C x 224 x 224
        output = (output * 255).to(device)
        return output

    final_states = preprocess(states[:, -1, :, :, :])
    demo_goals = preprocess([demo[-1] for demo in demos])

    with torch.no_grad():
        final_states_emb = vip(final_states)
        demo_goals_emb = vip(demo_goals)

    reward = torch.zeros(final_states_emb.shape[0])
    for demo in demo_goals_emb:
        reward -= torch.linalg.norm(final_states_emb - demo, dim=1).cpu()

    reward /= demo_goals_emb.shape[0]
    return reward

def vip_reward_trajectory_similarity(states, actions, **kwargs):
    """
    :param states: (K x T x H x W x C) np.ndarray entries in [0-1]
        K = batch size, T = trajectory size, nx = state embedding size

    :param actions: (K x T x nu) Tensor

    :returns rewards: (K x 1) Tensor
    """
    demos = kwargs['demos']

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

def reward_push_mug_right_to_left(state, action, **kwargs):
    ''' 
    task 94: pushing mug right to left
    :param state: (nx) np.ndarray
        output from tabletop_obs
    :param action: (T x nu) Tensor

    :returns scalar Tensor
    '''
    very_start = kwargs['very_start']
    left_to_right = -np.abs(state[3] - very_start[3] - 0.15) + 0.15
    drawer_move = np.abs(state[10] - very_start[10]) 
    move_faucet = state[12]

    penalty = 0 if (drawer_move < 0.03 and move_faucet < 0.01) else -100
    reward = left_to_right + penalty

    # for bad dynamics model
    if torch.isnan(torch.tensor([reward])):
        reward = -1e40

    return torch.Tensor([reward]).to(torch.float32)

def reward_push_mug_away(state, action, **kwargs):
    '''
    task 41: pushing mug away from camera
    :param state: (nx) np.ndarray
        output from tabletop_obs
    :param action: (T x nu) Tensor

    :returns scalar Tensor
    '''
    very_start = kwargs['very_start']
    x_shift = np.abs(state[3] - very_start[3])
    forward = -np.abs(state[4] - very_start[4] - 0.115) + 0.115

    penalty = 0 if (x_shift < 0.05) else -100
    reward = forward + penalty

    # for bad dynamics model
    if torch.isnan(torch.tensor([reward])):
        reward = -1e40

    return torch.Tensor([reward]).to(torch.float32)

def reward_close_drawer(state, action, **kwargs):
    '''
    task 5: closing drawer
    :param state: (K x nx) np.ndarray
        output from tabletop_obs
    :param action: (K x T x nu) Tensor

    :returns scalar Tensor
    '''
    very_start = kwargs['very_start']
    right_to_left = np.abs(state[3] - very_start[3])
    closed = state[10]

    penalty = 0 if right_to_left < 0.01 else -100
    reward = closed + penalty

    # for bad dynamics model
    if torch.isnan(torch.tensor([reward])):
        reward = -1e40

    return torch.Tensor([reward]).to(torch.float32)

def reward_turn_faucet_left_to_right(state, action, **kwargs):
    '''
    task 93: move something left to right
    '''
    move_faucet = -state[12]

    reward = move_faucet

    return torch.Tensor([reward]).to(torch.float32)

def reward_push_mug_close(state, action, **kwargs):
    '''
    task 44: pushing mug close to camera
    '''
    very_start = kwargs['very_start']
    reward = -np.abs(-state[4] + very_start[4] - 0.08) + 0.08

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

def get_engineered_reward(task_id):
    if task_id == 94:
        terminal_reward_fn = reward_push_mug_right_to_left
    elif task_id == 41:
        terminal_reward_fn = reward_push_mug_away
    elif task_id == 5:
        terminal_reward_fn = reward_close_drawer
    elif task_id == 93:
        terminal_reward_fn = reward_turn_faucet_left_to_right
    elif task_id == 44:
        terminal_reward_fn = reward_push_mug_close

    return terminal_reward_fn

def get_success_values(task_id, low_dim_state, very_start):
    if task_id == 94:
        rew = low_dim_state[3] - very_start[3]
        gt_reward = rew
        penalty = 0 # if (np.abs(low_dim_state[10] - very_start[10]) < 0.03 and low_dim_state[12] < 0.01) else -100
        success_threshold = 0.05
    elif task_id == 41:
        rew = low_dim_state[4] - very_start[4]
        gt_reward = -np.abs(rew - 0.115) + 0.115
        penalty = 0  # if (np.abs(low_dim_state[3] - very_start[3]) < 0.05) else -100
        success_threshold = 0.03
    elif task_id == 5:
        rew = low_dim_state[10]
        gt_reward = low_dim_state[10]
        penalty = 0 if (np.abs(low_dim_state[3] - very_start[3]) < 0.01) else -100
        success_threshold = -0.01
    elif task_id == 93:
        rew = -low_dim_state[12]
        gt_reward = rew
        penalty = 0
        success_threshold = 0.5
    elif task_id == 44:
        rew = - low_dim_state[4] + very_start[4]
        gt_reward = -np.abs(rew - 0.08) + 0.08
        penalty = 0 
        success_threshold = 0.04

    gt_reward += penalty
    succ = gt_reward > success_threshold

    return rew, gt_reward, succ
