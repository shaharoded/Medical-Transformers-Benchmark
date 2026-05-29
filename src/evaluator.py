from tqdm import tqdm
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import os
import pandas as pd

class Evaluator:
    def __init__(self, args):
        self.args = args

    def evaluate(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = '+split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0,num_samples,self.args.eval_batch_size),
                    desc='running forward pass')
        true, pred = [], []
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start+self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            true.append(batch['labels'])
            del batch['labels']
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            with torch.no_grad():
                pred.append(model(**batch).cpu())
        true, pred = torch.cat(true).numpy(), torch.cat(pred).numpy()
        if true.ndim==1:
            true = true.reshape(-1, 1)
            pred = pred.reshape(-1, 1)
        rows = []
        for i, outcome in enumerate(self.args.outcome_names):
            y_true, y_pred = true[:, i], pred[:, i]
            if len(np.unique(y_true))<2:
                rows.append({'outcome':outcome, 'auroc':np.nan, 'auprc':np.nan,
                             'minrp':np.nan, 'f1_0_5':np.nan, 'best_f1':np.nan,
                             'n_pos':int(y_true.sum()),
                             'n_neg':int((1-y_true).sum())})
                continue
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            f1_curve = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
            rows.append({'outcome':outcome,
                         'auroc':roc_auc_score(y_true, y_pred),
                         'auprc':average_precision_score(y_true, y_pred),
                         'minrp':np.minimum(precision, recall).max(),
                         'f1_0_5':f1_score(y_true, y_pred >= 0.5),
                         'best_f1':np.nanmax(f1_curve),
                         'n_pos':int(y_true.sum()),
                         'n_neg':int((1-y_true).sum())})
        table = pd.DataFrame(rows).set_index('outcome')
        result = {'auroc':float(table['auroc'].mean(skipna=True)),
                  'auprc':float(table['auprc'].mean(skipna=True)),
                  'minrp':float(table['minrp'].mean(skipna=True)),
                  'f1_0_5':float(table['f1_0_5'].mean(skipna=True)),
                  'best_f1':float(table['best_f1'].mean(skipna=True))}
        if split=='test':
            self.write_outputs(dataset, eval_ind, true, pred, table)
        if train_step is not None:
            self.args.logger.write('Result on '+split+' split at train step '
                              +str(train_step)+': '+str(result))
            self.args.logger.write('\nPer-outcome '+split+' results:\n'+str(table))
        return result

    def write_outputs(self, dataset, eval_ind, true, pred, table):
        table.to_csv(os.path.join(self.args.output_dir, 'test_per_outcome_metrics.csv'))
        patient_ids = [dataset.ind_to_ts_id[i] for i in eval_ind]
        pred_df = pd.DataFrame({'PatientId':patient_ids})
        for i, outcome in enumerate(self.args.outcome_names):
            pred_df['y_'+outcome] = true[:, i]
            pred_df['P_'+outcome] = pred[:, i]
            if dataset.first_hours is not None:
                pred_df['first_hour_'+outcome] = dataset.first_hours[eval_ind, i]
        pred_df.to_csv(os.path.join(self.args.output_dir, 'test_predictions.csv'), index=False)

        risk_rows = []
        input_hours = getattr(self.args, 'input_hours', 48.0)
        horizon_hours = getattr(self.args, 'horizon_hours', 288.0)
        for _, row in pred_df.iterrows():
            for hour in np.arange(input_hours+24.0, input_hours+horizon_hours+0.1, 24.0):
                out = {'PatientId':row['PatientId'], 'TimePoint':hour,
                       'IsInput':0, 'IsTerminal':0}
                for outcome in self.args.outcome_names:
                    out['P_'+outcome] = row['P_'+outcome]
                risk_rows.append(out)
        risk_df = pd.DataFrame(risk_rows)
        risk_df.to_csv(
            os.path.join(self.args.output_dir, 'test_risk_df.csv'), index=False)

