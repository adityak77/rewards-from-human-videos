import av
import torch
import numpy as np
import os
import h5py

from data_parser import WebmDataset
from data_augmentor import Augmentor
import torchvision
from transforms_video import *

import torchvision
from transforms_video import *

from collections import defaultdict, Counter
import json


FRAMERATE = 12  # default value

class VideoFolder(torch.utils.data.Dataset):

    def __init__(self, args, root, json_file_input, json_file_labels, clip_size,
                 nclips, step_size, is_val, num_tasks=174, transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 is_test=False):
        self.num_tasks = num_tasks
        self.is_val = is_val
        self.dataset_object = WebmDataset(args, json_file_input, json_file_labels,
                                      root, num_tasks=self.num_tasks, is_test=is_test, is_val=is_val)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.im_size = args.im_size
        self.batch_size = args.batch_size

        self.augmentor = Augmentor(augmentation_mappings_json,
                                   augmentation_types_todo)

        self.traj_length = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.similarity = args.similarity
        
        classes = []
        for key in self.classes_dict.keys():
            if not isinstance(key, int):
                classes.append(key)
        self.classes = classes
        num_occur = defaultdict(int)
        for c in self.classes:
            for video in self.json_data:
                if video.label == c:
                    num_occur[c] += 1
        if not self.is_val:
            with open(args.log_dir + '/human_data_tasks.txt', 'w') as f:
                json.dump(num_occur, f, indent=2)
        else:
            with open(args.log_dir + '/val_human_data_tasks.txt', 'w') as f:
                json.dump(num_occur, f, indent=2)
                
        # Every sample in batch: anchor (randomly selected class A), positive (randomly selected class A), 
        # and negative (randomly selected class not A)
        # Make dictionary for similarity triplets
        self.json_dict = defaultdict(list)
        for data in self.json_data:
            self.json_dict[data.label].append(data)
            
        print("Number of human videos: ", len(self.json_data), len(self.classes), "Total:", self.__len__())
        
        # Tasks used
        self.tasks = args.human_tasks
        assert(sum(num_occur.values()) == len(self.json_data))        
            
    def process_video(self, item):
         # Open video file
        try: 
            reader = av.open(item.path)
        except:
            print("Issue with opening the video, path:", item.path)
            assert(False)

        try:
            imgs = []
            imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
        except (RuntimeError, ZeroDivisionError) as exception:
            print('{}: WEBM reader cannot open {}. Empty '
                  'list returned.'.format(type(exception).__name__, item.path))
        orig_imgs = np.array(imgs).copy() 
        
        target_idx = self.classes_dict[item.label] 
        if not self.num_tasks == 174:
            target_idx = self.tasks.index(target_idx)
        
        imgs = self.transform_pre(imgs)
        imgs, label = self.augmentor(imgs, item.label)
        imgs = self.transform_post(imgs)
        
        num_frames = len(imgs)        
        if self.nclips > -1:
            num_frames_necessary = self.traj_length * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            offset = np.random.randint(0, diff)

        imgs = imgs[offset: num_frames_necessary + offset: self.step_size]
        if len(imgs) < (self.traj_length * self.nclips):
            imgs.extend([imgs[-1]] *
                        ((self.traj_length * self.nclips) - len(imgs)))

        # format data to torch
        data = torch.stack(imgs)
        data = data.permute(1, 0, 2, 3)
        return data
    
            
    def __getitem__(self, index):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """
            
        if self.similarity:
            # Need triplet for each sample
            item = random.choice(self.json_data) 
            
            # Get random anchor
            anchor = random.choice(self.json_dict[item.label])
            
            # Get negative 
            neg = random.choice(self.json_data)
            while neg.label == item.label:
                neg = random.choice(self.json_data)
                
            pos_data = self.process_video(item)  
            anchor_data  = self.process_video(anchor)
            neg_data = self.process_video(neg)
            return (pos_data, anchor_data, neg_data)
            

    def __len__(self):
        self.total_files = len(self.json_data)
        if self.similarity and not self.is_val and self.num_tasks <= 12:
            self.total_files = self.batch_size * 200 
        return self.total_files
