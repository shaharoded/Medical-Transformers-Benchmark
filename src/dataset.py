import pickle
import numpy as np
from tqdm import tqdm
import torch
from utils import CycleIndex
import os


class Dataset:
    def __init__(self, args) -> None:
        # read data
        filepath = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', args.dataset+'.pkl')
        )
        loaded = pickle.load(open(filepath,'rb'))
        if len(loaded)==6:
            data, oc, train_ids, val_ids, test_ids, metadata = loaded
        else:
            data, oc, train_ids, val_ids, test_ids = loaded
            metadata = {}
        args.outcome_names = metadata.get('outcome_names', ['in_hospital_mortality'])
        args.num_labels = len(args.outcome_names)
        args.static_varis = metadata.get('static_varis', None)
        args.input_hours = metadata.get('input_hours', 48.0)
        args.horizon_hours = metadata.get('horizon_hours', 288.0)
        # Length-of-stay regression target: z-scored hours from admission to RELEASE.
        # Patients who died or have no terminal event are masked out of the LoS loss.
        args.los_mean = float(metadata.get('los_mean', 0.0))
        args.los_std  = float(metadata.get('los_std',  1.0)) or 1.0
        self.args = args
        self.first_hours = None
        run, totalruns = list(map(int, args.run.split('o')))
        num_train = int(np.ceil(args.train_frac*len(train_ids)))
        start = int(np.linspace(0,len(train_ids)-num_train,totalruns)[run-1])
        train_ids = train_ids[start:start+num_train]
        num_val = int(np.ceil(args.train_frac*len(val_ids)))
        start = int(np.linspace(0,len(val_ids)-num_val,totalruns)[run-1])
        val_ids = val_ids[start:start+num_val]
        args.logger.write('\nPreparing dataset '+args.dataset)
        static_varis = self.get_static_varis(args.dataset)
        if args.dataset=='mimic_iii':
            # Filter labeled data in first 24h and fill missing age for old patients.
            data = data.loc[(data.minute>=0)&(data.minute<=24*60)]
            data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
            
        # keep variables seen in training set only
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(train_variables)]
        curr_ids = data.ts_id.unique()
        train_ids = np.intersect1d(train_ids, curr_ids)
        val_ids = np.intersect1d(val_ids, curr_ids)
        test_ids = np.intersect1d(test_ids, curr_ids)
        args.logger.write('# train, val, test TS: '+str([len(train_ids), len(val_ids), len(test_ids)]))
        sup_ts_ids = np.concatenate((train_ids, val_ids, test_ids))
        ts_id_to_ind = {ts_id:i for i,ts_id in enumerate(sup_ts_ids)}
        data = data.loc[data.ts_id.isin(sup_ts_ids)]
        data['ts_ind'] = data['ts_id'].map(ts_id_to_ind)

        # Get y and N
        oc = oc.loc[oc.ts_id.isin(sup_ts_ids)].copy()
        oc['ts_ind'] = oc['ts_id'].map(ts_id_to_ind)
        oc = oc.sort_values(by='ts_ind')
        y = np.array(oc[args.outcome_names]).astype(float)
        if y.ndim==1:
            y = y.reshape(-1, 1)
        first_hour_cols = [o+'__first_hour' for o in args.outcome_names]
        if all(c in oc.columns for c in first_hour_cols):
            self.first_hours = np.array(oc[first_hour_cols]).astype(float)
        # Length-of-stay target — raw hours (NaN where missing), plus a mask
        # and a z-scored copy aligned to the same patient ordering as `y`.
        if 'length_of_stay_hours' in oc.columns:
            los_raw = oc['length_of_stay_hours'].to_numpy(dtype=float)
            los_valid = np.isfinite(los_raw)
            self.los_hours_raw = los_raw
            self.los_mask = los_valid.astype(np.float32)
            los_filled = np.where(los_valid, los_raw, args.los_mean)
            self.los_target_norm = ((los_filled - args.los_mean) / args.los_std).astype(np.float32)
        else:
            self.los_hours_raw = np.full(len(oc), np.nan, dtype=float)
            self.los_mask = np.zeros(len(oc), dtype=np.float32)
            self.los_target_norm = np.zeros(len(oc), dtype=np.float32)
        N = len(sup_ts_ids)

        # To save
        self.N = N
        self.y = y
        self.static_varis = static_varis
        self.ind_to_ts_id = {i:ts_id for ts_id,i in ts_id_to_ind.items()}
        self.splits = {'train':[ts_id_to_ind[i] for i in train_ids],
                       'val':[ts_id_to_ind[i] for i in val_ids],
                       'test':[ts_id_to_ind[i] for i in test_ids]}
        self.splits['eval_train'] = self.splits['train'][:2000]
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)
        num_train = len(train_ids)
        num_train_pos = y[self.splits['train']].sum(axis=0)
        args.pos_class_weight = (num_train-num_train_pos)/np.maximum(num_train_pos, 1)
        args.logger.write('pos class weight: '+str(args.pos_class_weight))
        args.logger.write('% pos class in train, val, test splits: '
                          +str([num_train_pos/num_train,
                                y[self.splits['val']].sum(axis=0)/len(val_ids),
                                y[self.splits['test']].sum(axis=0)/len(test_ids)]))
        
        if 'llm' in args.model_type:
            self.data = data
            return
        
        # Get static data with missingness indicator.
        data = self.get_static_data(data)

        # Trim to max len.
        if args.model_type in ['strats', 'istrats']:
            data = data.sample(frac=1)
            data = data.groupby('ts_id').head(args.max_obs)
        elif args.model_type=='grud':
            timestamps = data[['ts_id','minute']].drop_duplicates().sample(frac=1)
            timestamps = timestamps.groupby('ts_id').head(args.max_timesteps)
            data = data.merge(timestamps, on=['ts_id','minute'], how='inner')

        # normalize if not aggregating, also get max_minute for strats
        args.finetune = args.load_ckpt_path is not None
        if args.finetune:
            pt_var_path = os.path.join(os.path.dirname(args.load_ckpt_path), 
                                       'pt_saved_variables.pkl')
            variables, means_stds, max_minute = pickle.load(open(pt_var_path,'rb'))
        if args.model_type in ['strats','istrats','grud']:
            if not(args.finetune):
                means_stds = data.loc[data.ts_id.isin(train_ids)].groupby(
                                    'variable').agg({'value':['mean', 'std']})
                means_stds.columns = [col[1] for col in means_stds.columns]
                means_stds.loc[means_stds['std']==0, 'std'] = 1
                max_minute = data['minute'].max()
            data = data.merge(means_stds.reset_index(), on='variable', how='left')
            data['value'] = (data['value']-data['mean'])/data['std']
            
        # prepare time series inputs
        if not(args.finetune):
            variables = data.variable.unique()
        var_to_ind = {v:i for i,v in enumerate(variables)}
        V = len(variables)
        args.V = V
        args.logger.write('# TS variables: '+str(V))
        if args.model_type in ['strats', 'istrats']:
            values = [[] for i in range(N)]
            times = [[] for i in range(N)]
            varis = [[] for i in range(N)]
            data['minute'] = data['minute']/max_minute*2-1
            for row in data.itertuples():
                values[row.ts_ind].append(row.value)
                times[row.ts_ind].append(row.minute)
                varis[row.ts_ind].append(var_to_ind[row.variable])
            self.values, self.times, self.varis = values, times, varis
        elif args.model_type=='grud':
            deltas = [[] for i in range(N)]
            values = [[] for i in range(N)]
            mask = [[] for i in range(N)]
            for ts_ind, curr_data in data.groupby('ts_ind'):
                curr_times = sorted(list(curr_data.minute.unique()))
                time2idx = {t:i for i,t in enumerate(curr_times)}
                T = len(curr_times)
                curr_values, curr_mask = np.zeros((T,V)),np.zeros((T,V))
                for row in curr_data.itertuples():
                    time_idx = time2idx[row.minute]
                    vind = var_to_ind[row.variable]
                    curr_values[time_idx, vind] = row.value
                    curr_mask[time_idx, vind] = 1
                curr_delta = np.zeros((T,V))
                for t in range(1,T):
                    curr_delta[t,:] = curr_times[t]-curr_times[t-1] \
                                    + (1-curr_mask[t-1])*curr_delta[t-1,:]
                deltas[ts_ind] = curr_delta/(24*60) # days
                values[ts_ind] = curr_values
                mask[ts_ind] = curr_mask
            self.values, self.mask = values, mask
            self.deltas = deltas
        
    def get_static_varis(self, dataset):
        if getattr(self.args, 'static_varis', None) is not None:
            static_varis = self.args.static_varis
        elif dataset=='mimic_iii':
            static_varis = ['Age', 'Gender']
        elif dataset=='physionet_2012':
            static_varis = ['Age', 'Gender', 'Height', 'ICUType_1',
                            'ICUType_2', 'ICUType_3', 'ICUType_4']
        elif dataset=='user_mimic_iv':
            static_varis = ['age_at_admission', 'gender', 'admission_type',
                            'has_diabetes_type1', 'has_diabetes_type2',
                            'has_hypertension', 'has_obesity']
        return static_varis

    def get_static_data(self, data):
                # Get static data with missingness indicator.
        static_ii = data.variable.isin(self.static_varis)
        static_data = data.loc[static_ii]
        data = data.loc[~static_ii] # remove static vars from data
        static_var_to_ind = {v:i for i,v in enumerate(self.static_varis)}
        D = len(static_var_to_ind)
        if self.args.dataset=='physionet_2012':
            D+=2
            self.static_varis += ['Gender_missing', 'Height_missing']
        demo = np.zeros((self.N, D))
        for row in tqdm(static_data.itertuples()):
            var_ind = static_var_to_ind[row.variable]
            demo[row.ts_ind, var_ind] = row.value
            if self.args.dataset=='physionet_2012':
                if row.variable=='Gender':
                    demo[row.ts_ind, D-2] = 1
                elif row.variable=='Height':
                    demo[row.ts_ind, D-1] = 1
        # mean fill missing static values
        if self.args.dataset=='physionet_2012':
            static_data_train = static_data.loc[static_data.ts_ind.isin(self.splits['train'])]
            gender_mean = static_data_train.loc[static_data_train.variable=='Gender']['value'].mean()
            height_mean = static_data_train.loc[static_data_train.variable=='Height']['value'].mean()
            del static_data_train
            gender_mask = (1-demo[:,D-2]).astype(bool)
            demo[gender_mask, static_var_to_ind['Gender']] = gender_mean
            height_mask = (1-demo[:,D-1]).astype(bool)
            demo[height_mask, static_var_to_ind['Height']] = height_mean
        # Normalize static data.
        train_ind = self.splits['train']
        means = demo[train_ind].mean(axis=0, keepdims=True)
        stds = demo[train_ind].std(axis=0, keepdims=True)
        stds = (stds==0) + (stds>0)*stds
        demo = (demo-means)/stds
        self.args.logger.write('# static features: '+str(D))
        # to save
        self.demo = demo
        self.args.D = D
        return data


    def get_batch(self, ind=None):
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
        if self.args.model_type in ['strats', 'istrats']:
            return self.get_batch_strats(ind)
        elif self.args.model_type=='grud':
            return self.get_batch_grud(ind)
        
        
    def get_batch_grud(self, ind):
        deltas = [self.deltas[i] for i in ind]
        values = [self.values[i] for i in ind]
        masks = [self.mask[i] for i in ind]
        num_timestamps = np.array(list(map(len, deltas)))
        max_timestamps = max(num_timestamps)
        pad_lens = max_timestamps-num_timestamps
        V = self.args.V
        pad_mats = [np.zeros((l,V)) for l in pad_lens]
        deltas = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0) 
                                    for delta,pad in zip(deltas,pad_mats)]))
        values = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0) 
                                    for delta,pad in zip(values,pad_mats)]))
        masks = torch.FloatTensor(np.stack([np.concatenate((delta,pad), axis=0) 
                                    for delta,pad in zip(masks,pad_mats)]))
        return {'delta_t':deltas, 'x_t':values, 'm_t':masks,
                'seq_len':torch.LongTensor(num_timestamps),
                'demo':torch.FloatTensor(self.demo[ind]),
                'labels':torch.FloatTensor(self.y[ind]),
                'los_target_norm':torch.FloatTensor(self.los_target_norm[ind]),
                'los_mask':torch.FloatTensor(self.los_mask[ind])}
    def get_batch_strats(self, ind):
        demo = torch.FloatTensor(self.demo[ind]) # N,D
        num_obs = [len(self.values[i]) for i in ind]
        max_obs = max(num_obs)
        pad_lens = max_obs-np.array(num_obs)
        values = [self.values[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        times = [self.times[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        varis = [self.varis[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        varis = torch.IntTensor(varis)
        obs_mask = [[1]*l1+[0]*l2 for l1,l2 in zip(num_obs,pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)
        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask, 'demo':demo,
                'labels':torch.FloatTensor(self.y[ind]),
                'los_target_norm':torch.FloatTensor(self.los_target_norm[ind]),
                'los_mask':torch.FloatTensor(self.los_mask[ind])}

        
            
            


