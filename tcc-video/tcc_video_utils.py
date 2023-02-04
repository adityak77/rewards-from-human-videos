
"""
Pytorch implementation of TCC based off https://github.com/google-research/google-research/tree/master/tcc/tcc
and https://github.com/June01/tcc_Temporal_Cycle_Consistency_Loss.pytorch.
"""

import torch
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _align_single_cycle(cycle, embs, cycle_length, num_steps, temperature):
    # choose from random frame
    n_idx = (torch.rand(1)*num_steps).long()[0]

    # Create labels
    labels = torch.tensor([n_idx], device=device)

    # Choose query feats for first frame.
    query_feats = embs[cycle[0], n_idx:n_idx + 1]
    num_channels = query_feats.size(-1)
    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c]]
        
        mean_squared_distance = torch.sum((query_feats.repeat([num_steps, 1]) -
                                            candidate_feats) ** 2, dim=1)
        similarity = -mean_squared_distance

        similarity /= float(num_channels)
        similarity /= temperature

        beta = F.softmax(similarity, dim=0).unsqueeze(1).repeat([1, num_channels])
        query_feats = torch.sum(beta * candidate_feats, dim=0, keepdim=True)

    return similarity, labels

def _align(cycles, embs, num_steps, num_cycles, cycle_length, temperature):
  """Align by finding cycles in embs."""
  logits_list = []
  labels_list = []
  for i in range(num_cycles):
    logits, labels = _align_single_cycle(cycles[i],
                                         embs,
                                         cycle_length,
                                         num_steps,
                                         temperature)
    logits_list.append(logits)
    labels_list.append(labels)

  logits = torch.stack(logits_list, dim=0)
  labels = torch.stack(labels_list, dim=0)

  return logits, labels

def gen_cycles(num_cycles, batch_size, cycle_length=2):
    """
    Generate random cycles for the stochastic alignment loss. We generate random
    cycles of length cycle_length and repeat them num_cycles times.

    param: num_cycles (int): number of cycles to generate
    param: batch_size (int): batch size
    param: cycle_length (int): length of each cycle
    """
    sorted_idxes = torch.arange(batch_size).unsqueeze(0).repeat([num_cycles, 1])
    sorted_idxes = sorted_idxes.view([batch_size, num_cycles])
    cycles = sorted_idxes[torch.randperm(len(sorted_idxes))].view([num_cycles, batch_size])
    cycles = cycles[:, :cycle_length]
    cycles = torch.cat([cycles, cycles[:, 0:1]], dim=1)

    return cycles

def regression_loss(logits, labels, steps, num_steps, variance_lambda=0.001):
    """
    Compute the regression loss for the stochastic alignment loss. We use the
    MSE loss with variance regularization.

    return: loss (torch.Tensor): loss
    """
    steps = steps.to(torch.float32)
    beta = F.softmax(logits, dim=1) # softmax last dimension

    # transform labels to start/end index labels in steps
    start_labels = torch.tensor([steps[i, labels[i], 0] for i in range(len(labels))], device=device)
    end_labels = torch.tensor([steps[i, labels[i], 0] for i in range(len(labels))], device=device)

    def time_loss(idx, true_time):
        # idx = 0 for start time, idx = 1 for end time
        pred_time = torch.sum(steps[:, :, idx] * beta, dim=1)

        # variance aware regression loss
        pred_time_tiled = pred_time.unsqueeze(1).repeat([1, num_steps])
        pred_time_variance = torch.sum(beta * (steps[:, :, idx] - pred_time_tiled) ** 2, dim=1)
        pred_time_log_var = torch.log(pred_time_variance)
        squared_error = (true_time - pred_time) ** 2
        return torch.mean(torch.exp(-pred_time_log_var) * squared_error + variance_lambda * pred_time_log_var)

    start_loss = time_loss(0, start_labels)
    end_loss = time_loss(1, end_labels)

    return start_loss + end_loss

def compute_stochastic_alignment_loss(embs,
                                      steps,
                                      num_steps,
                                      batch_size,
                                      num_cycles,
                                      cycle_length,
                                      temperature,
                                      variance_lambda):
    cycles = gen_cycles(num_cycles, batch_size, cycle_length)
    logits, labels = _align(cycles, embs, num_steps, num_cycles, cycle_length, temperature)

    # regression loss
    steps = steps[cycles[:, 0]]
    loss = regression_loss(logits, labels, steps, num_steps, variance_lambda)

    return loss

def compute_alignment_loss(embs,
                           batch_size,
                           steps=None,
                           seq_lens=None,
                           num_cycles=20,
                           cycle_length=2,
                           temperature=0.1,
                           variance_lambda=0.001):
    """
    Compute the TCC loss for a batch of videos. We use the best performing method in 
    the paper at https://arxiv.org/pdf/1904.07846.pdf. We do the regression MSE loss
    with variance regularization. We use L2 norm for the similarity.


    param: embs (torch.Tensor): frame embeddings of shape (batch_size, num_steps, emb_dim)
        typically num_steps = seq_lens * (seq_lens - 1) / 2
    param: batch_size (int): batch size
    param: steps (torch.Tensor): ground truth range of video embedding (batch_size, num_steps, 2)
    param: seq_lens (torch.Tensor): length of each video in the batch (batch_size)

    """
    embs = embs.to(device)

    num_steps = embs.shape[1]

    if not seq_lens:
        # assuming num_steps is approximately seq_lens * (seq_lens - 1) / 2
        video_lengths = int(1 + np.sqrt(1 + 8 * num_steps)) // 2
        seq_lens = torch.tensor(video_lengths, device=device).unsqueeze(0).repeat([batch_size])

    if not steps:
        steps_list = []
        for i in range(video_lengths):
            for j in range(i + 1, video_lengths):
                steps_list.append([i, j])
        steps = torch.tensor(steps_list, device=device).unsqueeze(0).repeat([batch_size, 1, 1])

    loss = compute_stochastic_alignment_loss(
        embs=embs,
        steps=steps,
        num_steps=num_steps,
        batch_size=batch_size,
        num_cycles=num_cycles,
        cycle_length=cycle_length,
        temperature=temperature,
        variance_lambda=variance_lambda
    )

    return loss