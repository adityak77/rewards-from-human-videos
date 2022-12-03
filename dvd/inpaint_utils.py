import os
import importlib

import numpy as np
import cv2
import torch

from tqdm import tqdm
from PIL import Image

import sys
sys.path.append('/home/akannan2/inpainting/detectron2')
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

sys.path.append('/home/akannan2/inpainting/E2FGVI/')
from core.utils import to_tensors

def get_human_cfg():
    config_file = '/home/akannan2/inpainting/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    weights = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    confidence_threshold = 0.5

    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

def get_robot_cfg():
    config_file = '/home/akannan2/inpainting/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    weights = '/home/akannan2/inpainting/detectron2/output/model_final.pth'
    confidence_threshold = 0.7
    
    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return cfg


def get_segmentation_model(cfg):
    predictor = DefaultPredictor(cfg)
    return predictor

def get_inpaint_model(args):
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

# def process_video(model, video, resolution=None):
#     count = 0
#     while video.isOpened() and count < 100:
#         success, frame = video.read()
#         if success:
#             count += 1
#             if resolution:
#                 frame = cv2.resize(frame, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
#             yield (frame, model(frame))
#         else:
#             break

def get_segmented_frames(video_frames, model, model_name, human_filter=False):
    resolution = None
    if model_name == 'e2fgvi_hq':
        height, width = video_frames[0].shape[0], video_frames[0].shape[1]
        downsample_resolution_factor = max(width // 400 + 1, height // 400 + 1)
        resolution = width // downsample_resolution_factor, height // downsample_resolution_factor

    if resolution:
        frames = [cv2.resize(frame, dsize=resolution, interpolation=cv2.INTER_CUBIC) for frame in video_frames]
    else:
        frames = video_frames

    frames_info = [model(frame) for frame in frames]

    masks = []
    for i in range(len(frames_info)):
        if not human_filter:
            masks.append(frames_info[i]['instances'].pred_masks)
        else:
            human_masks = []
            for j, cls_id in enumerate(frames_info[i]['instances'].pred_classes):
                if cls_id == 0:
                    hmask = frames_info[i]['instances'].pred_masks[j]
                    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    hmask_processed = cv2.dilate(hmask.cpu().numpy().astype(np.uint8), struct_element, iterations=5).astype(np.bool)
                    hmask_processed = torch.Tensor(hmask_processed).to(hmask.device)
                    human_masks.append(hmask_processed)

            if len(human_masks) == 0:
                masks.append(torch.Tensor(human_masks))
            else:
                masks.append(torch.stack(human_masks))
    
    return frames, masks

def inpaint(args, inpaint_model, segment_model, video_frames):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (args.width, args.height)
    else:
        size = None

    # prepare datset
    frames, masks = get_segmented_frames(video_frames, segment_model, args.model, human_filter=True)
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
            pred_imgs, _ = inpaint_model(masked_imgs, len(neighbor_ids))
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
    
    ret_frames = []
    for f in range(video_length):
        comp = comp_frames[f].astype(np.uint8)
        ret_frames.append(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))

    return ret_frames