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
import torchvision
import matplotlib.pyplot as plt
import random
import time

import importlib
import av
import argparse
import logging

from pytorch_mppi import mppi_metaworld as mppi

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

if __name__ == '__main__':
    TIMESTEPS = 51 # T = env.max_path_length
    N_SAMPLES = 100  # K
    NUM_ITERS = 10000

    ENGINEERED_REWARDS = args.engineered_rewards

    d = device
    dtype = torch.double

    u_min = torch.Tensor([-1.0, -1.0, -1.0, float('-inf')]).to(dtype=dtype)
    u_max = torch.Tensor([1.0, 1.0, 1.0, float('inf')]).to(dtype=dtype)

    noise_sigma = torch.eye(4, dtype=dtype, device=d) 
    lambda_ = 1e-2

    env = Tabletop(log_freq=args.env_log_freq, 
                   filepath=args.log_dir + '/env',
                   xml=args.xml,
                   verbose=args.verbose)  # bypass the default TimeLimit wrapper
    env.reset_model()

    video_encoder = load_encoder_model()
    sim_discriminator = load_discriminator_model()
    demo = torch.Tensor(decode_gif(args.demo_path))

    # using ground truth rewards
    if ENGINEERED_REWARDS:
        nx = 13 # env_info
        costs = running_cost_engineered
        terminal = None

        dynamics = no_dynamics
    else:
        # state space is image and need to translate back and forth from that
        nx = 64800 # size of the image 120 x 180 x 3
        costs = running_cost
        terminal = terminal_state_cost

        dynamics = no_dynamics

    logdir = 'engineered_reward' if ENGINEERED_REWARDS else args.checkpoint.split('/')[1]
    logdir = logdir + f'_{TIMESTEPS}_{N_SAMPLES}_{lambda_}'
    logdir = os.path.join('mppi_plots', logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    mppi_metaworld = mppi.MPPI(dynamics, costs, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                               terminal_state_cost=terminal, lambda_=lambda_, u_min=u_min, u_max=u_max)
    total_reward, total_successes, total_episodes, _ = mppi.run_mppi_metaworld(mppi_metaworld, env, train, args.task_id, terminal_state_cost,
                                                                               logdir, iter=NUM_ITERS, use_gt=ENGINEERED_REWARDS, render=False)
    # logger.info("Total reward %f", total_reward)
    logger.info(f"Fraction successful episodes: {total_successes} / {total_episodes} - {total_successes / total_episodes}")
