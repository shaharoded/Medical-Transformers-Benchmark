"""This file contain common utility functions."""
import argparse
from datetime import datetime
import string
import os
import random
import json
from pytz import timezone
from tqdm import tqdm
tqdm.pandas()
_local_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache"))
os.environ["HF_HOME"] = _local_hf_cache
os.environ["TRANSFORMERS_CACHE"] = _local_hf_cache
from transformers import set_seed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Optimizer
from typing import Any, Union


def get_curr_time() -> str:
    """Get current date and time in PST as str."""
    return datetime.now().astimezone(
            timezone('US/Pacific')).strftime("%d/%m/%Y %H:%M:%S")


class Logger: 
    """Class to write message to both output_dir/filename.txt and terminal."""
    def __init__(self, output_dir: str=None, filename: str=None) -> None:
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message: Any, show_time: bool=True) -> None:
        "write the message"
        message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith('\n'): 
                message = '\n'+get_curr_time()+' >> '+message[1:]
            else:
                message = get_curr_time()+' >> '+message
        print (message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message+'\n')


def set_all_seeds(seed: int) -> None:
    """Function to set seeds for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    set_seed(seed)


def count_parameters(logger: Logger, model: nn.Module):
    """Print model parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write('\nModel details:')
    logger.write('# parameters: '+str(total))
    logger.write('# trainable parameters: '+str(trainable)+', '
                 +str(100*trainable/total)+'%')

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    logger.write('#params by dtype:')
    for k, v in dtypes.items():
        logger.write(str(k)+': '+str(v)+', '+str(100*v/total)+'%')


class TimeSeriesModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        if args.model_type!='istrats':
            self.demo_emb = nn.Sequential(nn.Linear(args.D, args.hid_dim*2),
                                          nn.Tanh(),
                                          nn.Linear(args.hid_dim*2, args.hid_dim))
        if args.model_type=='istrats':
            ts_demo_emb_size = args.hid_dim+args.D
        else:
            ts_demo_emb_size = args.hid_dim*2
        # The prediction head emits a single vector of length (num_labels + 1):
        # the first num_labels logits are the per-outcome binary risk scores
        # (BCE-with-logits), and the last scalar is the z-scored length-of-stay
        # regression (MSE against the normalised target during training,
        # denormalised in the evaluator to report MAE in hours).
        head_out_dim = args.num_labels + 1
        self.finetune = args.load_ckpt_path is not None
        if self.finetune:
            self.forecast_head = nn.Linear(ts_demo_emb_size, args.V)
            self.binary_head = nn.Linear(args.V, head_out_dim)
        else:
            self.binary_head = nn.Linear(ts_demo_emb_size, head_out_dim)
        self.register_buffer('pos_class_weight', torch.as_tensor(args.pos_class_weight).float())
        self.los_loss_weight = float(getattr(args, 'los_loss_weight', 1.0))

    def binary_cls_final(self, logits, labels, los_target_norm=None, los_mask=None):
        """
        Combined head:
          - first K columns of `logits` -> binary outcome BCE
          - last column of `logits`    -> z-scored LoS regression (MSE,
            masked to RELEASE-discharged patients).

        Training: returns scalar loss = BCE + los_loss_weight * masked-MSE.
        Eval:     returns tensor of shape [bsz, K+1] where the first K
                  columns are sigmoid-probabilities and the last column is
                  the *normalised* LoS prediction. The evaluator denormalises
                  the last column (multiply by los_std, add los_mean) before
                  computing MAE in hours.
        """
        K = self.args.num_labels
        binary_logits = logits[:, :K]
        los_pred_norm = logits[:, K]
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                binary_logits, labels, pos_weight=self.pos_class_weight)
            if los_target_norm is not None and los_mask is not None:
                # Masked MSE on z-scored LoS; only RELEASE-discharged patients
                # contribute. Division by max(mask.sum(), 1) keeps the gradient
                # scale stable on batches with few/no valid LoS samples.
                sq_err = (los_pred_norm - los_target_norm) ** 2 * los_mask
                denom = torch.clamp(los_mask.sum(), min=1.0)
                loss = loss + self.los_loss_weight * sq_err.sum() / denom
            return loss
        # Eval: probabilities for the K binary outcomes + normalised LoS.
        binary_probs = torch.sigmoid(binary_logits)
        return torch.cat([binary_probs, los_pred_norm.unsqueeze(-1)], dim=-1)



class CycleIndex:
    """Class to generate batches of training ids, 
    shuffled after each epoch.""" 
    def __init__(self, indices:Union[int,list], batch_size: int,
                 shuffle: bool=True) -> None:
        if type(indices)==int:
            indices = np.arange(indices)
        self.indices = indices
        self.num_samples = len(indices)
        self.batch_size = batch_size
        self.pointer = 0
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle

    def get_batch_ind(self):
        """Get indices for next batch."""
        start, end = self.pointer, self.pointer + self.batch_size
        # If we have a full batch within this epoch, then get it.
        if end <= self.num_samples:
            if end==self.num_samples:
                self.pointer = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
            else:
                self.pointer = end
            return self.indices[start:end]
        # Otherwise, fill the batch with samples from next epoch.
        last_batch_indices_incomplete = self.indices[start:]
        remaining = self.batch_size - (self.num_samples-start)
        self.pointer = remaining
        if self.shuffle:
            np.random.shuffle(self.indices)
        return np.concatenate((last_batch_indices_incomplete, 
                               self.indices[:remaining]))
    



