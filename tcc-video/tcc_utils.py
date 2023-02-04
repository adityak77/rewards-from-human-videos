
"""
Pytorch implementation of TCC based off https://github.com/google-research/google-research/tree/master/tcc/tcc
and https://github.com/June01/tcc_Temporal_Cycle_Consistency_Loss.pytorch.
"""

import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _align_single_cycle(cycle, embs, cycle_length, num_steps, temperature):
    # choose from random frame
    n_idx = (torch.rand(1)*num_steps).long()[0]

    # Create labels
    onehot_labels = torch.eye(num_steps)[n_idx].to(device)

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

    return similarity, onehot_labels

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

def regression_loss(logits, labels, num_steps, steps, seq_lens, variance_lambda=0.001):
    """
    Compute the regression loss for the stochastic alignment loss. We use the
    MSE loss with variance regularization.

    return: loss (torch.Tensor): loss
    """
    steps = steps.to(torch.float32)
    beta = F.softmax(logits, dim=1) # softmax last dimension

    true_time = torch.sum(steps * labels, dim=1)
    pred_time = torch.sum(steps * beta, dim=1)

    # variance aware regression loss
    pred_time_tiled = pred_time.unsqueeze(1).repeat([1, num_steps])
    pred_time_variance = torch.sum(beta * (steps - pred_time_tiled) ** 2, dim=1)

    pred_time_log_var = torch.log(pred_time_variance)
    squared_error = (true_time - pred_time) ** 2
    return torch.mean(torch.exp(-pred_time_log_var) * squared_error + variance_lambda * pred_time_log_var)

def compute_stochastic_alignment_loss(embs,
                                      steps,
                                      seq_lens,
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
    seq_lens = seq_lens[cycles[:, 0]]
    loss = regression_loss(logits, labels, num_steps, steps, seq_lens, variance_lambda)

    return loss

def compute_alignment_loss(embs,
                           batch_size,
                           num_cycles=20,
                           cycle_length=2,
                           temperature=0.1,
                           variance_lambda=0.001):
    """
    Compute the TCC loss for a batch of videos. We use the best performing method in 
    the paper at https://arxiv.org/pdf/1904.07846.pdf. We do the regression MSE loss
    with variance regularization. We use L2 norm for the similarity.


    param: embs (torch.Tensor): frame embeddings of shape (batch_size, num_frames, emb_dim)
    param: batch_size (int): batch size

    """
    embs = embs.to(device)

    num_steps = embs.shape[1]

    steps = torch.arange(0, num_steps, device=device).unsqueeze(0).repeat([batch_size, 1])
    seq_lens = torch.tensor(num_steps, device=device).unsqueeze(0).repeat([batch_size])

    loss = compute_stochastic_alignment_loss(
        embs=embs,
        steps=steps,
        seq_lens=seq_lens,
        num_steps=num_steps,
        batch_size=batch_size,
        num_cycles=num_cycles,
        cycle_length=cycle_length,
        temperature=temperature,
        variance_lambda=variance_lambda
    )

    return loss