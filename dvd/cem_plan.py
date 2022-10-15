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

from pytorch_mppi.mppi_metaworld import tabletop_obs, evaluate_episode


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--task_id", type=int, default=94, help="Task to learn a model for")
parser.add_argument("--demo_path", type=str, default='data_correct/demo_sample0.gif', help='path to demo video')
parser.add_argument("--checkpoint", type=str, default='test/tasks6_seed0_lr0.01_sim_pre_hum54144469394_dem60_rob54193/model/150sim_discriminator.pth.tar', help='path to model')

parser.add_argument('--similarity', action='store_true', default=True, help='whether to use similarity discriminator') # needs to be true for MultiColumn init
parser.add_argument('--hidden_size', type=int, default=512, help='latent encoding size')
parser.add_argument('--num_tasks', type=int, default=2, help='number of tasks') # needs to exist for MultiColumn init
parser.add_argument('--im_size', type=int, default=120, help='image size to process by encoder')

# parser.add_argument("--action_dim", type=int, default=5)
# parser.add_argument("--replay_buffer_size", type=int, default=100) 
parser.add_argument("--log_dir", type=str, default="mppi")
parser.add_argument("--env_log_freq", type=int, default=20) 
parser.add_argument("--verbose", type=bool, default=1) 
# parser.add_argument("--num_epochs", type=int, default=100)
# parser.add_argument("--traj_length", type=int, default=20)
# parser.add_argument("--num_traj_per_epoch", type=int, default=3) 
# parser.add_argument("--random", action='store_true', default=False) # take random actions
# parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--xml", type=str, default='env1')

parser.add_argument("--engineered_rewards", action='store_true', default=False, help='Use hand engineered rewards or not')

args = parser.parse_args()

"""     Functions for inferring trajectory reward      """
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
    :param states: (K x T x nx) Tensor
        T = trajectory size, nx = state embedding size 
    :param demo: (T x H x W x C) Tensor
    """
    states = torch.reshape(states.squeeze(), (states.shape[1], states.shape[2], 120, 180, 3))
    states = (states.numpy() * 255).astype(np.uint8)

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
    def process_video(samples):
        """
        :param samples: (K x T x H x W x C) Tensor
            to prepare before encoding by networks
        """
        K, T, H, W, C = tuple(samples.shape)

        samples = torch.transpose(samples, 0, 1) # shape T x K x H x W x C
        sample_downsample = samples[1::max(1, len(samples) // 30)][:30]
        sample_downsample = torch.transpose(sample_downsample, 0, 1) # shape T x K x H x W x C
        # Flatten as (K x T) x H x W x C
        sample_flattened = torch.reshape(sample_downsample, (-1, H, W, C))
        
        sample_flattened = [elem for elem in sample_flattened] # need to convert to list to make PIL image conversion

        # sample_transform should be K x 30 x T x 120 x 120 after cropping and trajectory downsampling
        sample_transform = torch.stack(transform(sample_flattened))
        sample_transform = sample_transform.reshape(K, 30, C, 120, 120).permute(0, 2, 1, 3, 4)
        sample_data = [sample_transform.to(device)]

        return sample_data
    
    demo_data = process_video(demo.unsqueeze(0))
    states_data = process_video(torch.Tensor(states))

    # evaluate trajectory reward
    with torch.no_grad():
        states_enc = model.encode(states_data) # K x 512
        demo_enc = model.encode(demo_data) # 1 x 512

        logits = sim_discriminator.forward(states_enc, demo_enc.repeat(states_enc.shape[0], 1))
        reward_samples = F.softmax(logits, dim=1)

    return reward_samples[:, 1].unsqueeze(0)


"""     Inputs for MPPI      """
def no_dynamics(state, action):
    """
    unknown dynamics
    might be very inefficient
    """
    return state

def running_cost(state, action):
    return torch.zeros(state.shape[0])

def running_cost_engineered(state, action):
    # task 94: pushing mug left to right
    left_to_right = -state[:, 3]

    return left_to_right.to(torch.float32)
    
def terminal_state_cost(states, actions):
    """
    :param states: (K x T x nx) Tensor
        K = batch size, T = trajectory size, nx = state embedding size

    :param actions: (K x T x nu) Tensor

    :returns costs: (K x 1) Tensor
    """
    command_start = time.perf_counter()
    rewards = inference(states, demo, video_encoder, sim_discriminator)
    elapsed = time.perf_counter() - command_start
    print(f"Video similarity inference elapsed time: {elapsed:.4f}s")

    return -rewards

def train(new_data):
    pass

def running_reward_engineered(state, action):
    # task 94: pushing mug right to left
    left_to_right = state[:, 3]
    # penalty = (left_to_right > 0.1).to(torch.float) * -100
    reward = left_to_right # + penalty

    return reward.to(torch.float32)

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

    # for logging
    logdir = 'engineered_reward'
    logdir = logdir + f'_{TIMESTEPS}_{N_SAMPLES}_{NUM_EPISODES}_sampling_rewards'
    logdir = os.path.join('cem_plots', logdir)
    logdir_episode = os.path.join(logdir, 'episodes')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    if not os.path.isdir(logdir_episode):
        os.makedirs(logdir_episode)

    total_successes = 0
    total_episodes = 0
    rolling_success = deque()
    dvd_reward_history = []
    episode_reward_history = []
    succ_rate_history = []

    cumulative_step_rewards = []
    for ep in range(NUM_EPISODES):
        done = False
        iters = 0
        step_rewards = []
        mean_sampling_rewards = []
        while not done:
            tstart = time.perf_counter()
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
            
            mean_sampling_rewards.append(sample_rewards.mean() + sum(step_rewards))
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
            traj_distribution = MultivariateNormal(actions_mean, actions_cov)
            traj_samples = traj_distribution.sample((1,))
            action = traj_samples[0, iters*nu:(iters+1)*nu] # from CEM
            s, r, done, low_dim_info = env.step(action.cpu().numpy())
            iters += 1
            tend = time.perf_counter()
            rew = tabletop_obs(low_dim_info)[3] - very_start[3]
            step_rewards.append(rew)
            logger.debug(f"{env.cur_path_length}: Step time: {tend-tstart:.4f}\t Step reward: {rew:.6f}\t Action: {action}")

        cumulative_step_rewards.append(step_rewards)
        # calculate success and reward of trajectory
        low_dim_state = tabletop_obs(low_dim_info)
        episode_reward = low_dim_state[3] - very_start[3]
        # penalty = (episode_reward > 0.1) * -100
        # episode_reward += penalty

        succ = episode_reward > 0.05 # and episode_reward < 0.1
        
        # print results
        result = 'SUCCESS' if succ else 'FAILURE'
        total_successes += succ
        total_episodes += 1
        rolling_success.append(succ)
        print(f'----------Episode done: {result} | episode_reward: {episode_reward}----------')
        print(f'----------Currently at {total_successes} / {total_episodes}----------')
        
        if len(rolling_success) > 10:
            rolling_success.popleft()
        
        succ_rate = sum(rolling_success) / len(rolling_success)

        # episode_reward_history.append(episode_reward)
        episode_reward_history.append(sum(step_rewards))
        succ_rate_history.append(succ_rate)

        # logging results
        with open(os.path.join(logdir, 'episode_results.pkl'), 'wb') as f:
            pickle.dump(cumulative_step_rewards, f)

        print('----------REPLOTTING----------')
        plt.figure()
        plt.plot([i for i in range(len(episode_reward_history))], episode_reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward in episode')
        plt.savefig(os.path.join(logdir, 'engineered_reward_episode.png'))
        plt.close()
        
        plt.figure()
        plt.plot([i for i in range(len(succ_rate_history))], succ_rate_history)
        plt.xlabel('Episode')
        plt.ylabel('Rolling Success Rate')
        plt.savefig(os.path.join(logdir, 'rolling_success_rate_episode.png'))
        plt.close()

        plt.figure()
        plt.plot([i for i in range(len(step_rewards))], step_rewards)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title(f'State rewards in episode {ep}')
        plt.savefig(os.path.join(logdir_episode, f'episode_rewards{ep}.png'))
        plt.close()
        
        plt.figure()
        plt.plot([i for i in range(len(mean_sampling_rewards))], mean_sampling_rewards)
        plt.xlabel('Step')
        plt.ylabel('Mean Sampled Trajectories Reward')
        plt.title(f'Mean Reward of Sampled Trajectories in episode {ep}')
        plt.savefig(os.path.join(logdir_episode, f'episode_sampled_traj_rewards{ep}.png'))
        plt.close()

        _, start_info = env.reset_model()
        very_start = tabletop_obs(start_info)