import os
import sys
import signal
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import (load_args, remove_module_from_checkpoint_state_dict, 
                   setup_cuda_devices, save_checkpoint, setup_ddp, cleanup_ddp)
from callbacks import (PlotLearning, AverageMeter)
from multi_column import MultiColumn, SimilarityDiscriminator
import torchvision
from transforms_video import ComposeMix, RandomCropVideo, RandomRotationVideo, Scale
from data_loader_av import VideoFolder
import cv2
import imageio
import pickle
import json
from PIL import Image

from similarity import train_similarity, validate_similarity 


def main():
    # load args
    args = load_args()

    # setup device - CPU or GPU
    dev, device_ids = setup_cuda_devices(args)
    print(" > Using device: {}".format(dev.type))
    print(" > Active GPU ids: {}".format(device_ids))

    args.human_tasks = [int(i) for i in args.human_tasks]
    args.robot_tasks = [int(i) for i in args.robot_tasks]
    args.num_tasks = len(args.human_tasks)
    if args.just_robot:
        args.num_tasks = len(args.robot_tasks)
        args.human_tasks = args.robot_tasks
    # set run output folder
    save_dir = args.log_dir + 'tasks' + str(args.num_tasks) + '_seed' + str(args.seed) + '_lr' + str(args.lr)
    if args.traj_length != 0:
        save_dir +='_traj' + str(args.traj_length)
    if args.similarity:
        save_dir += '_sim'
    if args.pretrained:
        save_dir += '_pre'
    save_dir += '_hum'
    for num in args.human_tasks:
        save_dir += str(num) 
    if args.add_demos:
        save_dir += '_dem' + str(args.add_demos) + '_rob'
        for num in args.robot_tasks:
            save_dir += str(num)
    if args.im_size != 120:
        save_dir += '_im' + str(args.im_size)
    if args.just_robot:
        save_dir += '_justrobot'
    if args.inpaint:
        save_dir += '_inpainted_egohos_hq'
    if args.sd_augment:
        save_dir += '_sd_augment'
    if args.lang_template:
        save_dir += '_lang_template'
    if args.lang_label:
        save_dir += '_lang_label'
    if args.lang_align:
        save_dir += '_lang_align'
    
    print(" > Output folder for this run -- {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))
        os.makedirs(os.path.join(save_dir, 'model'))
    with open(save_dir + '/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.log_dir = save_dir

    mp.spawn(training_loop, args=(args,), nprocs=num_gpus, join=True)


def training_loop(rank, args):
    print(f"Running training loop on device {rank}")
    # device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(rank)

    # set column model
    cnn_def = importlib.import_module("{}".format('model3D_1'))

    best_loss = float('Inf')

    setup_ddp(rank, args.num_gpus)
    dev0 = (rank * 2) % args.num_gpus
    dev1 = (rank * 2 + 1) % args.num_gpus

    # create model
    print(" > Creating model ... !")
    model = MultiColumn(args, args.num_tasks, cnn_def.Model,
                        int(args.hidden_size), dev0=dev0, dev1=dev1)

    if args.resume or args.pretrained: # optionally resume from a checkpoint
        if args.pretrained:
            checkpoint_path = os.path.join(args.pretrained_dir, "model_best.pth.tar")
        else:
            checkpoint_path = os.path.join(args.log_dir, 'model',
                                   str(args.resume) + 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            if args.pretrained:
                print("Loading in pretrained model")
                checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
                                          checkpoint['state_dict'])
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif args.resume:
                print("Loading in model from resume")
                checkpoint['encoder_state_dict'] = remove_module_from_checkpoint_state_dict(
                                                checkpoint['encoder_state_dict'])
                model.load_state_dict(checkpoint['encoder_state_dict'])
                best_loss = checkpoint['best_loss']

            start_epoch = checkpoint['epoch']
            print(" > Loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print(" !#! No checkpoint found at '{}'".format(
                checkpoint_path))
            assert(False)
    # model = model.to(device)
    # model = DDP(model) 
        
    if args.similarity:
        sim_discriminator = SimilarityDiscriminator(args).to(dev0)
        if args.sim_resume:
            resume_path = os.path.join(args.log_dir, 'model', str(args.sim_resume) + 'sim_discriminator.pth.tar')
            sim_discriminator.load_state_dict(torch.load(resume_path), strict=True)
        # sim_discriminator = DDP(sim_discriminator, device_ids=[dev0])
    if args.pretrained and not args.lang_align:
        for p in model.parameters():
            p.requires_grad = False
            
    print("Trainable params in encoder:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # define augmentation pipeline
    upscale_size_train = int(args.im_size * 1.4)
    upscale_size_eval = int(args.im_size * 1)

    # Random crop videos during training
    transform_train_pre = ComposeMix([
            [RandomRotationVideo(15), "vid"],
            [Scale(upscale_size_train), "img"],
            [RandomCropVideo(args.im_size), "vid"],
             ])

    # Center crop videos during evaluation
    transform_eval_pre = ComposeMix([
            [Scale(upscale_size_eval), "img"],
            [torchvision.transforms.ToPILImage(), "img"],
            [torchvision.transforms.CenterCrop(args.im_size), "img"],
             ])

    # Transforms common to train and eval sets and applied after "pre" transforms
    transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Normalize(
                       mean=[0.485, 0.456, 0.406],  # default values for imagenet
                       std=[0.229, 0.224, 0.225]), "img"]
             ])
    
    # Transform for robot demos
    robot_demo_transform = ComposeMix([
        [RandomRotationVideo(15), "vid"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(args.im_size), "img"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
         ])        

    train_data = VideoFolder(args,
                             root=args.human_data_dir,
                             json_file_input=args.json_data_train,
                             json_file_labels=args.json_file_labels,
                             clip_size=args.traj_length,
                             nclips=1,
                             step_size=1,
                             num_tasks=args.num_tasks,
                             is_val=False,
                             transform_pre=transform_train_pre,
                             transform_post=transform_post,
                             robot_demo_transform=robot_demo_transform,
                             )

    num_dataloader_workers = 6 * args.num_gpus
    print(" > Using {} processes for data loader.".format(num_dataloader_workers))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=num_dataloader_workers, pin_memory=True,
        drop_last=True, sampler=DistributedSampler(train_data))

    val_data = VideoFolder(args, 
                           root=args.human_data_dir,
                           json_file_input=args.json_data_val,
                           json_file_labels=args.json_file_labels,
                           clip_size=args.traj_length,
                           nclips=1,
                           step_size=1,
                           num_tasks=args.num_tasks,
                           is_val=True,
                           transform_pre=transform_eval_pre,
                           transform_post=transform_post,
                           robot_demo_transform=robot_demo_transform,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=num_dataloader_workers, pin_memory=True,
        drop_last=True, sampler=DistributedSampler(val_data))

    print(" > Number of dataset classes : {}".format(len(train_data.classes_dict.keys())//2))
    assert len(train_data.classes_dict.keys())//2 == args.num_tasks

    # define loss function (criterion)
    if args.lang_label or args.lang_template:
        loss_class = nn.MSELoss().to(dev0) # .to(device)
    else:
        loss_class = nn.CrossEntropyLoss().to(dev0) # .to(device)

    # define optimizer
    lr = args.lr
    last_lr = 1e-05
    params = list(model.parameters())
    if args.similarity:
        params += list(sim_discriminator.parameters())
        print("Number of discriminator params", sum(p.numel() for p in sim_discriminator.parameters() if p.requires_grad))

        optimizer = torch.optim.SGD(params, lr,
                                 momentum=0.9,
                                 weight_decay=0.00001)
        
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # set callbacks
    plotter = PlotLearning(args, os.path.join(
        args.log_dir, "plots"), args.num_tasks)
    lr_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 'min', factor=0.5, patience=5, verbose=True)
    if args.resume:
        lr_decayer.load_state_dict(checkpoint['lr_decayer_state_dict'])
        plotter = checkpoint['plotter']
    val_loss = float('Inf')

    print(" > Training is getting started...")
    print(" > Training takes {} epochs.".format(args.num_epochs))
    start_epoch = args.resume + 1 if args.resume > 0 else 0
    
    if args.resume:
        report_losses = checkpoint['report_losses']
    else:
        report_losses = {}
        report_losses['train_acc'] = []
        report_losses['val_acc'] = []
        report_losses['val_loss'] = []
        report_losses['train_loss'] = []
        report_losses['false_pos'] = []
        report_losses['false_neg'] = []
        report_losses['false_pos_train'] = []
        report_losses['false_neg_train'] = []

    try:
        video_encoder = model.module
    except:
        video_encoder = model

    try:
        video_encoder = model.module
    except:
        video_encoder = model

    for epoch in range(start_epoch, args.num_epochs):
        lrs = [params['lr'] for params in optimizer.param_groups]
        print(" > Current LR(s) -- {}".format(lrs))
        if np.max(lr) < last_lr and last_lr > 0:
            print(" > Training is DONE by learning rate {}".format(last_lr))
            sys.exit(1)

        if args.similarity:
            train_loader.sampler.set_epoch(epoch)
            train_loss, train_top1, class_loss, false_pos_train, false_neg_train = train_similarity(args, 
            train_loader, model, sim_discriminator, loss_class, optimizer, epoch, dev0)
        
        # evaluate on validation set
        if epoch % args.log_freq == 0:
            print("Evaluating on epoch", epoch)
            if args.similarity:
                val_loader.sampler.set_epoch(epoch)
                val_loss, val_top1, false_pos, false_neg = validate_similarity(args, val_loader, model, 
                sim_discriminator, loss_class, epoch, dev0)
                
            # set learning rate
            lr_decayer.step(val_loss)

            # plot learning
            plotter_dict = {}
            plotter_dict['loss'] = train_loss
            plotter_dict['val_loss'] = 0 
            plotter_dict['class_loss'] = class_loss
            plotter_dict['val_acc'] = val_top1 
            plotter_dict['learning_rate'] = lr
            plotter_dict['false_pos_train'] = false_pos_train
            plotter_dict['false_neg_train'] = false_neg_train
            plotter_dict['false_pos'] = false_pos
            plotter_dict['false_neg'] = false_neg
            plotter_dict['val_loss'] = val_loss
            plotter_dict['acc'] = train_top1
            plotter_dict['val_acc'] = val_top1
            
            plotter.plot(plotter_dict)
            
            report_losses['val_acc'].append(val_top1)
            report_losses['train_acc'].append(train_top1)
            report_losses['val_loss'].append(val_loss)
            np.savetxt(args.log_dir + '/val_acc.txt', np.array(report_losses['val_acc']), fmt='%f')
            np.savetxt(args.log_dir + '/train_acc.txt', np.array(report_losses['train_acc']), fmt='%f')
            np.savetxt(args.log_dir + '/val_loss.txt', np.array(report_losses['val_loss']), fmt='%f')
            if args.similarity:
                report_losses['false_pos'].append(false_pos)
                report_losses['false_neg'].append(false_neg)
                report_losses['false_pos_train'].append(false_pos_train)
                report_losses['false_neg_train'].append(false_neg_train)
                np.savetxt(args.log_dir + '/false_pos.txt', np.array(report_losses['false_pos']), fmt='%f')
                np.savetxt(args.log_dir + '/false_neg.txt', np.array(report_losses['false_neg']), fmt='%f')
                np.savetxt(args.log_dir + '/false_pos_train.txt', np.array(report_losses['false_pos_train']), fmt='%f')
                np.savetxt(args.log_dir + '/false_neg_train.txt', np.array(report_losses['false_neg_train']), fmt='%f')
                
            print(" > Validation accuracy after epoch {} = {}".format(epoch, val_top1))

            # remember best loss and save the checkpoint
            freq = 10 if args.similarity else 5
            if (epoch + 1) % freq == 0:
                if args.similarity and rank == 0:
                    is_best = val_loss < best_loss
                    best_loss = min(val_loss, best_loss)

                    save_state = {
                        'epoch': epoch + 1,
                        'best_loss': best_loss,
                        'encoder_state_dict': model.state_dict(),
                        'sim_discriminator_state_dict': sim_discriminator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_decayer_state_dict': lr_decayer.state_dict(),
                        'report_losses': report_losses,
                        'plotter': plotter,
                    }
                    save_dir = os.path.join(args.log_dir, 'model')
                    save_checkpoint(save_state, is_best, save_dir, str(epoch+1) + 'checkpoint.pth.tar')
    
    cleanup_ddp()
            

if __name__ == '__main__':
    main()
