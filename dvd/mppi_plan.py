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

from pytorch_mppi import mppi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

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
    :param states: (T x nx) Tensor
        T = trajectory size, nx = state embedding size 
    :param demo: (T x nx) Tensor
    """
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
    def process_video(sample):
        sample_downsample = sample[1::max(1, len(sample) // 30)][:30]
        sample_downsample = [elem for elem in sample_downsample] # need to convert to list to make PIL image conversion

        sample_transform = torch.stack(transform(sample_downsample)).permute(1, 0, 2, 3).unsqueeze(0)
        sample_data = [sample_transform.to(device)]

        return sample_data
    
    states_data = process_video(states)
    demo_data = process_video(demo)

    # evaluate trajectory reward
    with torch.no_grad():
        states_enc = model.encode(states_data)
        demo_enc = model.encode(demo_data)

        input_demo = F.softmax(sim_discriminator.forward(states_enc, demo_enc), dim=1)

    reward_sample = input_demo[:, 1].item()

    return reward_sample


"""     Inputs for MPPI      """
def dynamics(state, action):
    """
    unknown dynamics
    not sure if this will work
    """
    return state

def running_cost(state, action):
    return torch.zeros(state.shape[0])

def terminal_state_cost(states, actions):
    """
    :param states: (K x T x nx) Tensor
        K = batch size, T = trajectory size, nx = state embedding size

    :param actions: (K x T x nu) Tensor

    :returns costs: (K x 1) Tensor
    """
    video_encoder = load_encoder_model()
    sim_discriminator = load_discriminator_model()
    demo = decode_gif(args.demo_path)

    costs = torch.zeros(states.shape[0])
    for i in range(states.shape[0]):
        costs[i] = inference(states[i], demo, video_encoder, sim_discriminator)

    return costs

def train(new_data):
    pass

if __name__ == '__main__':
    TIMESTEPS = 60  # T
    N_SAMPLES = 200  # K

    d = device
    dtype = torch.double

    noise_sigma = 10 * torch.eye(4, dtype=dtype, device=d) 
    # noise_sigma = torch.tensor(10, device=d, dtype=dtype)
    # noise_sigma = torch.tensor([[10, 0], [0, 10]], device=d, dtype=dtype)
    lambda_ = 1.

    env = Tabletop(log_freq=args.env_log_freq, 
                   filepath=args.log_dir + '/env',
                   xml=args.xml,
                   verbose=args.verbose)  # bypass the default TimeLimit wrapper
    env.reset_model()

    # TODO: state space is image and need you translate back and forth from that
    nx = 7 # 1024 # idk for now
    mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         terminal_state_cost=terminal_state_cost, lambda_=lambda_)
    total_reward = mppi.run_mppi(mppi_gym, env, train, render=True)
    logger.info("Total reward %f", total_reward[0])
