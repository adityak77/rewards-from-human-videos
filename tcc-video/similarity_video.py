import time

import torch
import numpy as np

from utils import *
from callbacks import AverageMeter
from transforms_video import *
from tcc_video_utils import compute_alignment_loss, compute_hard_nearest_neighbor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chunk_and_encode(args, model, data, traj_length, min_chunk_length):
    chunks = []
    for sample in data:
        # sample : (3, traj_length, H, W)
        for i in range(traj_length):
            for j in range(i + 1 + min_chunk_length, traj_length + 1):
                chunks.append(sample[:, i:j].to(device))

    # chunks : (args.batch_size * K) x 3 x traj_length x H x W
    import ipdb; ipdb.set_trace()
    chunk_enc = model.encode(chunks) # chunk_enc : (args.batch_size * K) x 512
    chunk_enc = chunk_enc.reshape(args.batch_size, -1, 512) # (args.batch_size, K, 512)

    return chunk_enc

def train_video_similarity(args, train_loader, model, sim_discriminator, loss_class, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    enc_losses = AverageMeter()
    top1 = AverageMeter()
    
    class_losses = AverageMeter()
    false_pos_meter = AverageMeter()
    false_neg_meter = AverageMeter()
    torch.autograd.set_detect_anomaly(True)

    # switch to train mode
    model.train()
    sim_discriminator.train()

    # random length video trajectories
    if args.traj_length == 0:
        rand_length = np.random.randint(20, 40)
        train_loader.dataset.traj_length = rand_length
        print("training rand length:", train_loader.dataset.traj_length)
    
    len_dataloader = len(train_loader)
    data_source_iter = iter(train_loader)
    i = 0
    while i < len_dataloader:
        start = time.time()
        data_source = data_source_iter.next()
        pos_data, anchor_data, neg_data = data_source

        data_time.update(time.time() - start)
        
        model.zero_grad()
        sim_discriminator.zero_grad()

        # TODO: prepare data into chunks
        min_chunk_length = np.random.randint(10, rand_length)
        pos_chunk_enc = chunk_and_encode(args, model, pos_data, rand_length, min_chunk_length)
        anchor_chunk_enc = chunk_and_encode(args, model, anchor_data, rand_length, min_chunk_length)
        neg_chunk_enc = chunk_and_encode(args, model, neg_data, rand_length, min_chunk_length)
        
        # TODO: generate TCC loss
        input_enc = torch.cat([pos_chunk_enc, anchor_chunk_enc], dim=0).to(device) # (2 * args.batch_size, K, 512)

        seq_lens = torch.tensor(rand_length, device=device).unsqueeze(0).repeat([2 * args.batch_size])

        steps_list = []
        for i in range(rand_length):
            for j in range(i + min_chunk_length, rand_length):
                steps_list.append([i, j])
        steps = torch.tensor(steps_list, device=device).unsqueeze(0).repeat([2 * args.batch_size, 1, 1])
        enc_loss = compute_alignment_loss(input_enc, steps, seq_lens, num_cycles=args.batch_size)

        optimizer.zero_grad()
        enc_loss.backward()
        optimizer.step()

        # TODO: Match anchor chunks with nearest neighbor in pos/neg chunks
        # all encodings are (args.batch_size * K, 512)
        pos_enc = compute_hard_nearest_neighbor(anchor_chunk_enc, pos_chunk_enc)
        neg_enc = compute_hard_nearest_neighbor(anchor_chunk_enc, neg_chunk_enc)
        anchor_enc = torch.flatten(anchor_chunk_enc, start_dim=0, end_dim=1)

        pos_enc = pos_enc.detach()
        neg_enc = neg_enc.detach()
        anchor_enc = anchor_enc.detach()
        
        # Calculate loss for sim discriminator
        pos_anchor = sim_discriminator.forward(pos_enc, anchor_enc)
        neg_anchor = sim_discriminator.forward(anchor_enc, neg_enc)
        pos_anchor_label = torch.ones(pos_anchor.shape[0]).long().cuda()
        neg_anchor_label = torch.zeros(neg_anchor.shape[0]).long().cuda()
        class_out = torch.cat((pos_anchor, neg_anchor))  
        sim_labels = torch.cat((pos_anchor_label, neg_anchor_label))
        class_loss = loss_class(class_out, sim_labels)

        loss = enc_loss + class_loss # or not optimize jointly?

        # measure accuracy and record loss
        prec1 = float(((class_out[:, 0] < class_out[:, 1]).to(torch.long) == sim_labels[:]).sum()) / class_out.shape[0]
        top1.update(prec1, 1) #class_out.size(0)
        losses.update(loss.item(), 1)
        enc_losses.update(enc_loss.item(), 1)
        class_losses.update(class_loss.item(), 1)
        false_pos = float(((class_out[:, 0] < class_out[:, 1]) & (sim_labels[:] == 0)).sum()) / float((sim_labels[:] == 0).sum())
        false_neg = float(((class_out[:, 0] > class_out[:, 1]) & (sim_labels[:] == 1)).sum()) / float((sim_labels[:] == 1).sum())
        false_pos_meter.update(false_pos, 1)
        false_neg_meter.update(false_neg, 1)

        # compute gradient and do SGD step for task classifier
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - start)

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Encoder Loss {enc_loss.val:.4f} ({enc_loss.avg:.4f})\t'
                  'Classloss {class_losses.val:.3f} ({class_losses.avg:.3f})\t'.format(
                      epoch, i, len_dataloader, batch_time=batch_time,
                      data_time=data_time, top1=top1, loss=losses, enc_loss=enc_losses, class_losses=class_losses))
        i += 1
    return losses.avg, top1.avg, enc_losses.avg, class_losses.avg, false_pos_meter.avg, false_neg_meter.avg

def validate_video_similarity(args, val_loader, model, sim_discriminator, loss_class, epoch):
    pass