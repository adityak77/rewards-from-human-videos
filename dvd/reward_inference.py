import os
import argparse
import importlib

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from multi_column import MultiColumn, SimilarityDiscriminator

from callbacks import AverageMeter
from transforms_video import *
import imageio
import av
from PIL import Image, ImageSequence

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

    input_encs = []

    for demo, input in test_data:
        def process_one(sample):
            sample_downsample = sample[1::max(1, len(sample) // 30)][:30]
            sample_downsample = [elem for elem in sample_downsample] # need to convert to list to make PIL image conversion

            sample_transform = torch.stack(transform(sample_downsample)).permute(1, 0, 2, 3).unsqueeze(0)
            sample_data = [sample_transform.to(device)]

            return sample_data
        input_data = process_one(input)
        demo_data = process_one(demo)

        with torch.no_grad():
            input_enc = model.encode(input_data)
            demo_enc = model.encode(demo_data)

            input_demo = F.softmax(sim_discriminator.forward(input_enc, demo_enc), dim=1)
            print('Reward sample:', input_demo[0, 1].item())

        reward_sample = input_demo[:, 1].item()
        reward.update(reward_sample)

        input_encs.append(input_demo)

    return reward.avg

def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def prepare_results(args):
    def decode_gif(path):
        try: 
            reader = av.open(path)
        except:
            print("Issue with opening the video, path:", path)
            assert(False)

        return [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]

    print('Demo:', args.demo_path)
    demo_gif = decode_gif(args.demo_path)
    eval_paths = [os.path.join(args.eval_path, 'env', fpath) for fpath in os.listdir(os.path.join(args.eval_path, 'env')) if fpath.endswith('.gif') or fpath.endswith('.mp4')]

    print('Input Samples', eval_paths)
    test_data = [(demo_gif, decode_gif(fpath)) for fpath in eval_paths]

    print("Loading in pretrained model")
    cnn_def = importlib.import_module("{}".format('model3D_1'))
    model = MultiColumn(args, args.num_tasks, cnn_def.Model, int(args.hidden_size))
    model_checkpoint = os.path.join('pretrained/video_encoder/', 'model_best.pth.tar')
    model.load_state_dict(remove_module_from_checkpoint_state_dict(torch.load(model_checkpoint)['state_dict']), strict=False)

    sim_discriminator = SimilarityDiscriminator(args)
    sim_discriminator.load_state_dict(torch.load(args.checkpoint), strict=True)

    reward = inference(test_data, model, sim_discriminator)

    print(f'Average Reward: {reward}')


if __name__ == '__main__':
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True, help='path to directory of test videos')
    parser.add_argument("--demo_path", type=str, default='data_correct/demo_sample0.gif', help='path to demo video')
    parser.add_argument("--checkpoint", type=str, default='test/tasks6_seed0_lr0.01_sim_pre_hum54144469394_dem60_rob54193/model/150sim_discriminator.pth.tar', help='path to model')

    parser.add_argument('--similarity', action='store_true', default=True, help='whether to use similarity discriminator') # needs to be true for MultiColumn init
    parser.add_argument('--hidden_size', type=int, default=512, help='latent encoding size')
    parser.add_argument('--num_tasks', type=int, default=2, help='number of tasks') # needs to exist for MultiColumn init
    parser.add_argument('--im_size', type=int, default=120, help='image size to process by encoder')

    args = parser.parse_args()

    prepare_results(args)