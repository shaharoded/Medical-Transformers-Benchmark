import torch.nn as nn
import argparse
from utils import Logger
import torch
import torch.nn.functional as F


def count_parameters(logger: Logger, model: nn.Module):
    """Print no. of parameters in model, no. of traininable parameters,
     no. of parameters in each dtype."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write('\nModel details:')
    logger.write('# parameters: '+str(total))
    logger.write('# trainable parameters: '+str(trainable)+', '\
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
        self.finetune = args.load_ckpt_path is not None
        if self.finetune:
            self.forecast_head = nn.Linear(ts_demo_emb_size, args.V)
            self.binary_head = nn.Linear(args.V,args.num_labels)
            self.register_buffer('pos_class_weight', torch.as_tensor(args.pos_class_weight).float())
        else:
            self.binary_head = nn.Linear(ts_demo_emb_size,args.num_labels)
            self.register_buffer('pos_class_weight', torch.as_tensor(args.pos_class_weight).float())

    def binary_cls_final(self, logits, labels):
        if labels is not None:
            return F.binary_cross_entropy_with_logits(logits, labels, 
                                    pos_weight=self.pos_class_weight)
        else:
            return F.sigmoid(logits)
