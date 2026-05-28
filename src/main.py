import argparse
import os
from utils import Logger, set_all_seeds
import torch
from dataset import Dataset
from modeling_strats import Strats
import numpy as np
from tqdm import tqdm
from transformers.optimization import AdamW
from models import count_parameters
from evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument('--dataset', type=str, default='user_mimic_iv')
    parser.add_argument('--train_frac', type=float, default=0.5)
    parser.add_argument('--run', type=str, default='1o10')

    # model related arguments
    parser.add_argument('--model_type', type=str, default='strats',
                        choices=['strats','istrats'])
    parser.add_argument('--load_ckpt_path', type=str, default=None)
    ##  strats and istrats
    parser.add_argument('--max_obs', type=int, default=880)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)
    # training/eval realated arguments
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--print_train_loss_every', type=int, default=100)
    parser.add_argument('--validate_after', type=int, default=-1)
    parser.add_argument('--validate_every', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()
    return args


def set_output_dir(args: argparse.Namespace) -> None:
    """Function to automatically set output dir 
    if it is not passed in args."""
    if args.output_dir is None:
        if args.load_ckpt_path is not None:
            args.output_dir_prefix = 'finetune_'+args.output_dir_prefix
        args.output_dir = '../outputs/'+args.dataset+'/'+args.output_dir_prefix
        args.output_dir += args.model_type
        for param in ['num_layers', 'hid_dim', 'num_heads', 'dropout', 'attention_dropout', 'lr']:
            args.output_dir += ','+param+':'+str(getattr(args, param))
        for param in ['train_frac', 'run']:
            args.output_dir += '|'+param+':'+str(getattr(args, param))
    os.makedirs(args.output_dir, exist_ok=True)




if __name__ == "__main__":
    # Preliminary setup.
    args = parse_args()
    set_output_dir(args)
    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n'+str(args))
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    set_all_seeds(args.seed+int(args.run.split('o')[0]))
    model_path_best = os.path.join(args.output_dir, 'checkpoint_best.bin')

    # load data
    dataset = Dataset(args)

    # load model
    model = Strats(args)
    model.to(args.device)
    count_parameters(args.logger, model)
    if args.load_ckpt_path is not None:
        curr_state_dict = model.state_dict()
        pt_state_dict = torch.load(args.load_ckpt_path)
        for k,v in pt_state_dict.items():
            if k in curr_state_dict:
                curr_state_dict[k] = v
        model.load_state_dict(curr_state_dict)

    # training loop
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = num_train/args.train_batch_size
    args.logger.write('\nNo. of training batches per epoch = '
                      +str(num_batches_per_epoch))
    args.max_steps = int(round(num_batches_per_epoch)*args.max_epochs)
    if args.validate_every is None:
        args.validate_every = int(np.ceil(num_batches_per_epoch))
    cum_train_loss, num_steps, num_batches_trained = 0,0,0
    wait, patience_reached = args.patience, False
    best_val_metric  = -np.inf
    best_val_res, best_test_res = None, None
    optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    train_bar = tqdm(range(args.max_steps))
    evaluator = Evaluator(args)

    # results before any training
    if args.validate_after<0:
        results = evaluator.evaluate(model, dataset, 'val',  train_step=-1)
        evaluator.evaluate(model, dataset, 'eval_train', train_step=-1)
        evaluator.evaluate(model, dataset, 'test', train_step=-1)
    
    model.train()
    for step in train_bar:
        # load batch
        batch = dataset.get_batch()
        batch = {k:v.to(args.device) for k,v in batch.items()}

        # forward pass
        loss = model(**batch)

        # backward pass
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.3)
            if (step+1)%args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

        # add to cum loss
        cum_train_loss += loss.item()
        num_steps += 1
        num_batches_trained += 1

        # Log training losses.
        train_bar.set_description(str(np.round(cum_train_loss/num_batches_trained,5)))
        if (num_steps)%args.print_train_loss_every == 0:
            args.logger.write('\nTrain-loss at step '+str(num_steps)+': '
                              +str(cum_train_loss/num_batches_trained))
            cum_train_loss, num_batches_trained = 0, 0

        # run validatation
        if (num_steps>=args.validate_after) and (num_steps%args.validate_every==0):
            # get metrics on test and validation splits
            val_res = evaluator.evaluate(model, dataset, 'val', train_step=step)
            evaluator.evaluate(model, dataset, 'eval_train', train_step=step)
            test_res = evaluator.evaluate(model, dataset, 'test', train_step=step)
            model.train(True)

            # Save ckpt if there is an improvement.
            curr_val_metric = val_res['auprc']+val_res['auroc']
            if curr_val_metric>best_val_metric:
                best_val_metric = curr_val_metric
                best_val_res, best_test_res = val_res, test_res
                args.logger.write('\nSaving ckpt at ' + model_path_best)
                torch.save(model.state_dict(), model_path_best)
                wait = args.patience
            else:
                wait -= 1
                args.logger.write('Updating wait to '+str(wait))
                if wait==0:
                    args.logger.write('Patience reached')
                    break
    
    # print final res
    args.logger.write('Final val res: '+str(best_val_res))
    args.logger.write('Final test res: '+str(best_test_res))
