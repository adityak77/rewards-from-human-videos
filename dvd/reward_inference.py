import os
import argparse

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from multi_column import MultiColumn, SimilarityDiscriminator

from callbacks import AverageMeter
from transforms_video import *
import imageio
from PIL import Image, ImageSequence

from model3D_1 import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

def inference(test_data, model, sim_discriminator):
    reward = AverageMeter()

    transform = ComposeMix([
        [Scale(args.im_size), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(args.im_size), "img"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
         ])

    model = model.to(device)
    sim_discriminator = sim_discriminator.to(device)

    model.eval()
    sim_discriminator.eval()

    for input, anchor in test_data:
        input_downsample = input[1::max(1, len(input) // 30)][:40]
        input_downsample = [elem for elem in input_downsample] # need to convert to list to make PIL image conversion

        anchor_downsample = input[1::max(1, len(anchor) // 30)][:40]
        anchor_downsample = [elem for elem in anchor_downsample] # need to convert to list to make PIL image conversion

        input_transform = torch.stack(transform(input_downsample)).permute(1, 0, 2, 3).unsqueeze(0)
        anchor_transform = torch.stack(transform(anchor_downsample)).permute(1, 0, 2, 3).unsqueeze(0)
        input_data = [input_transform.to(device)]
        anchor_data = [anchor_transform.to(device)]

        input_enc = model.encode(input_data)
        anchor_enc = model.encode(anchor_data)

        input_anchor = F.softmax(sim_discriminator.forward(input_enc, anchor_enc))
        print(sim_discriminator.forward(input_enc, anchor_enc))

        # import ipdb; ipdb.set_trace()
        reward_sample = input_anchor[:, 1].item()
        reward.update(reward_sample)

    return reward.avg

def prepare_results(args):
    def decode_gif(path):
        return np.array(imageio.mimread(path))[:, :, :, :3]

    anchor_gif = decode_gif(args.anchor_path)
    eval_paths = [os.path.join(args.eval_path, 'env', fpath) for fpath in os.listdir(os.path.join(args.eval_path, 'env')) if fpath.endswith('.gif')]

    print(eval_paths)
    test_data = [(anchor_gif, decode_gif(fpath)) for fpath in eval_paths]

    model = MultiColumn(args, args.num_tasks, Model, int(args.hidden_size))
    model_checkpoint = os.path.join('pretrained/video_encoder/', 'model_best.pth.tar')
    model.load_state_dict(torch.load(model_checkpoint)['state_dict'], strict=False)

    sim_discriminator = SimilarityDiscriminator(args)
    sim_discriminator.load_state_dict(torch.load(args.checkpoint), strict=True)

    reward = inference(test_data, model, sim_discriminator)

    print(f'Average Reward: {reward}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True, help='path to directory of test videos')
    parser.add_argument("--anchor_path", type=str, default='data_correct/demo_sample0.gif', help='path to demo video')
    parser.add_argument("--checkpoint", type=str, default='test/tasks6_seed0_lr0.01_sim_pre_hum54144469394_dem60_rob54193/model/150sim_discriminator.pth.tar', help='path to model')

    parser.add_argument('--similarity', action='store_true', default=True, help='whether to use similarity discriminator') # needs to be true for MultiColumn init
    parser.add_argument('--hidden_size', type=int, default=512, help='latent encoding size')
    parser.add_argument('--num_tasks', type=int, default=2, help='number of tasks') # needs to exist for MultiColumn init
    parser.add_argument('--im_size', type=int, default=120, help='image size to process by encoder')

    args = parser.parse_args()

    prepare_results(args)