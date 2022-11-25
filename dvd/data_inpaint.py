import os
import json
import numpy as np
import argparse
import random

from collections import namedtuple
from collections import defaultdict

from tqdm import tqdm
import torch
import cv2
import importlib
from PIL import Image

import sys
sys.path.append('/home/akannan2/inpainting/E2FGVI/')
# from test import resize_frames, process_masks
from core.utils import to_tensors
from segment_video import get_segmented_frames

ListData = namedtuple('ListData', ['id', 'label', 'path'])


class DatasetBase(object):
    """
    To read json data and construct a list containing video sample `ids`,
    `label` and `path`
    """
    def __init__(self, args, json_path_input, json_path_labels, data_root,
                 extension, num_tasks, is_test=False, is_val=False):
        self.num_tasks = num_tasks
        self.json_path_input = json_path_input
        self.json_path_labels = json_path_labels
        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test
        self.is_val = is_val
        
        self.num_occur = defaultdict(int)
        
        self.tasks = args.human_tasks

        # preparing data and class dictionary
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()
        print("Number of human videos:", self.num_occur.values())
        
        
    def read_json_input(self):
        json_data = []
        with open(self.json_path_input, 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                label = self.clean_template(elem['template'])
                if label not in self.classes_dict.keys(): # or label == 'Pushing something so that it slightly moves':
                    continue
                if label not in self.classes:
                    raise ValueError("Label mismatch! Please correct")
                
                item = ListData(elem['id'],
                                label,
                                os.path.join(self.data_root,
                                                elem['id'] + self.extension)
                                )
                json_data.append(item)
                self.num_occur[label] += 1

        return json_data

    def read_json_labels(self):
        classes = []
        with open(self.json_path_labels, 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)

    def get_two_way_dict(self, classes):
        classes_dict = {} 
        tasks = self.tasks
        for i, item in enumerate(classes):
            if i not in tasks:
                continue
            classes_dict[item] = i
            classes_dict[i] = item
        print("Length of keys", len(classes_dict.keys()), classes_dict.keys())
        return classes_dict

    def clean_template(self, template):
        """ Replaces instances of `[something]` --> `something`"""
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template

class WebmDataset(DatasetBase):
    def __init__(self, args, json_path_input, json_path_labels, data_root, num_tasks, 
                 is_test=False, is_val=False):
        EXTENSION = ".webm"
        super().__init__(args, json_path_input, json_path_labels, data_root,
                         EXTENSION, num_tasks, is_test, is_val)

def get_model(args):
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    return model

def get_ref_index(f, neighbor_ids, length, ref_length, num_ref):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index

# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size

def process_masks(masks, size = None):
    masks_expanded = []
    for mask in masks:
        if mask.shape[0] == 0:
            m = np.zeros(size).astype(np.uint8)
        else:
            m = np.clip(mask.cpu().numpy().astype(np.uint8).sum(axis=0), 0, 1)

        m = Image.fromarray(np.uint8(m), mode='L')
        m = m.resize(size, Image.NEAREST)

        m = np.array(m)
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks_expanded.append(Image.fromarray(m * 255))

    return masks_expanded


def inpaint(args, model, video_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (args.width, args.height)
    else:
        size = None

    # prepare datset
    args.use_mp4 = True if video_path.endswith('.mp4') or video_path.endswith('.gif') or video_path.endswith('.webm') else False
    input_ext = None
    if args.use_mp4:
        input_ext = '.' + video_path.split('.')[-1]
    # print(
    #     f'Loading videos and masks from: {video_path} | INPUT mp4/gif format: {args.use_mp4}'
    # )
    frames, masks = get_segmented_frames(args, video_path, human_filter=True)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode='RGB') for frame in frames]

    frames, size = resize_frames(frames, size)
    h, w = size[1], size[0]
    video_length = len(frames)
    imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    masks = process_masks(masks, size)
    binary_masks = [
        np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
    ]
    masks = to_tensors()(masks).unsqueeze(0)
    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None] * video_length

    # completing holes by e2fgvi
    # print(f'Start test...')
    for f in tqdm(range(0, video_length, args.neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - args.neighbor_stride),
                             min(video_length, f + args.neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_length=args.step, num_ref=args.num_ref)
        selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - selected_masks)
            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_imgs[i]).astype(
                    np.uint8) * binary_masks[idx] + frames[idx] * (
                        1 - binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32) * 0.5 + img.astype(np.float32) * 0.5

    # saving videos
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"VP90"), # for webm extension
                             args.savefps, size)
    for f in range(video_length):
        comp = comp_frames[f].astype(np.uint8)
        writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    writer.release()


def get_args():
    parser = argparse.ArgumentParser(description='Inpaint Something-Something videos for certain tasks')
    parser.add_argument('--human_data_dir', type=str, help='dir to human data', required=True)
    parser.add_argument('--root', type=str, default='./', help='root dir') 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--human_tasks', nargs='*', default=[5, 41, 93], help='if using robot demos, which tasks to include robot demos for')

    # E2FGVI args
    parser.add_argument("-c", "--ckpt", type=str, default='/home/akannan2/inpainting/E2FGVI/release_model/E2FGVI-CVPR22.pth')
    parser.add_argument("--model", type=str, default='e2fgvi', choices=['e2fgvi', 'e2fgvi_hq'])
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--num_ref", type=int, default=-1)
    parser.add_argument("--neighbor_stride", type=int, default=1)
    parser.add_argument("--savefps", type=int, default=24)

    # args for e2fgvi_hq (which can handle videos with arbitrary resolution)
    parser.add_argument("--set_size", action='store_true', default=False)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)

    # Detectron2 args
    parser.add_argument(
        "--config-file",
        default="/home/akannan2/inpainting/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.json_data_train = args.root + "something-something-v2-train.json"
    args.json_data_val = args.root + "something-something-v2-validation.json"
    args.json_data_test = args.root + "something-something-v2-test.json"
    args.json_file_labels = args.root + "something-something-v2-labels.json"
    random.seed(args.seed)
    args.human_tasks = list(map(int, args.human_tasks))
    
    return args


def main(args):
    root = os.path.join(args.human_data_dir, '20bn-something-something-v2')
    json_file_input = args.json_data_val # args.json_data_train
    json_file_labels = args.json_file_labels
    tasks = args.human_tasks

    # capture all videos with the same tasks
    print(f'Inpainting images for tasks {tasks}')
    dataset_object = WebmDataset(args, json_file_input, json_file_labels, root, len(tasks))

    json_data = dataset_object.json_data

    output_path = os.path.join(args.human_data_dir, '20bn-something-something-v2-inpainted')
    
    os.makedirs(output_path, exist_ok=True)

    # get model
    model = get_model(args)

    # inpaint using model
    for item in tqdm(json_data):
        # inpaint video at item.path and output it with same name in output_path
        filename = os.path.basename(item.path)
        save_path = os.path.join(output_path, filename)

        inpaint(args, model, item.path, save_path)

if __name__ == '__main__':
    # python data_inpaint.py --human_data_dir ../smthsmth/sm2/ --human_tasks 5 41 44 46 93 94 \\
    # --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
    args = get_args()
    main(args)