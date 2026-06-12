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


class PretrainDataset(Dataset):
    """Forecasting-pretraining dataset for STraTS.

    Ports `dataset_pretrain.py` from the official Tipirneni & Reddy
    implementation, adapted to our two-pickle preprocess (`*_pretrain.pkl`
    carries the full 0-336 h trajectory; the supervised input window is
    *not* applied here). Train+val patients are pooled; test patients are
    held out completely so the forecast head never sees them.

    For each sample drawn at training time:
      - pick a random anchor `t_anchor` from this patient's set of unique
        observation timestamps that are >= `min_anchor_minutes` (default 12 h);
      - take the last up-to-`max_obs` triplets that fall in
        `(t_anchor - obs_window_minutes, t_anchor]` as the encoder input;
      - target = the most-recent observed value of each variable inside
        `(t_anchor, t_anchor + forecast_window_minutes]` (default 2 h).
    Loss is masked MSE over variables that were actually observed in the
    forecast window (handled by `TimeSeriesModel.forecast_final`).
    """

    def __init__(self, args) -> None:
        # Read the dedicated pretrain pickle. Schema mirrors the supervised
        # pickle but `oc` carries no usable labels (we don't pretrain against
        # labels) so we tolerate either form.
        from config import (
            PRETRAIN_OBS_WINDOW_HOURS,
            PRETRAIN_FORECAST_WINDOW_MIN,
            PRETRAIN_MIN_ANCHOR_HOURS,
            PRETRAIN_MAX_DATA_HOURS,
            PRETRAIN_MAX_OBS,
        )

        filepath = self._resolve_pretrain_path(args)
        loaded = pickle.load(open(filepath, 'rb'))
        if len(loaded) == 6:
            data, _oc, train_ids, val_ids, test_ids, metadata = loaded
        else:
            data, _oc, train_ids, val_ids, test_ids = loaded
            metadata = {}
        args.outcome_names = metadata.get('outcome_names', [])
        args.num_labels = len(args.outcome_names)
        args.static_varis = metadata.get('static_varis', None)
        args.input_hours = metadata.get('input_hours', PRETRAIN_OBS_WINDOW_HOURS)
        args.horizon_hours = metadata.get('horizon_hours', PRETRAIN_MAX_DATA_HOURS - PRETRAIN_OBS_WINDOW_HOURS)
        args.los_mean = 0.0
        args.los_std  = 1.0
        # Pretraining never touches labels -> dummy LoS weight so downstream
        # code that probes `args.pos_class_weight` doesn't fault.
        args.pos_class_weight = np.zeros(max(args.num_labels, 1))

        self.args = args
        self.first_hours = None
        self.max_obs = PRETRAIN_MAX_OBS
        self.obs_window_minutes      = PRETRAIN_OBS_WINDOW_HOURS * 60.0
        self.forecast_window_minutes = float(PRETRAIN_FORECAST_WINDOW_MIN)
        self.min_anchor_minutes      = PRETRAIN_MIN_ANCHOR_HOURS * 60.0
        self.max_minute              = PRETRAIN_MAX_DATA_HOURS * 60.0
        args.max_obs = self.max_obs

        # Confine trajectory to the pretrain window. Test patients are
        # excluded entirely; the forecast head must never see test data,
        # so finetuning measures supervised lift cleanly.
        data = data.loc[(data.minute >= 0) & (data.minute <= self.max_minute)]
        data = data.loc[~data.ts_id.isin(test_ids)]

        args.logger.write('\nPreparing pretrain dataset '+args.dataset)
        static_varis = self.get_static_varis(args.dataset)

        # Variable vocabulary derived from training patients only (to avoid
        # leaking val-only variables into the encoder). Drop anything else.
        train_variables = data.loc[data.ts_id.isin(train_ids)].variable.unique()
        all_variables = data.variable.unique()
        delete_variables = np.setdiff1d(all_variables, train_variables)
        args.logger.write('Removing variables not in training set: '+str(delete_variables))
        data = data.loc[data.variable.isin(train_variables)]
        val_ids = np.intersect1d(val_ids, data.ts_id.unique())
        train_ids = np.intersect1d(train_ids, data.ts_id.unique())

        unsup_ts_ids = np.concatenate((train_ids, val_ids))
        ts_id_to_ind = {ts_id: i for i, ts_id in enumerate(unsup_ts_ids)}
        data = data.loc[data.ts_id.isin(unsup_ts_ids)].copy()
        data['ts_ind'] = data['ts_id'].map(ts_id_to_ind)
        N = len(unsup_ts_ids)

        args.logger.write('# train, val TS (pretrain): '+str([len(train_ids), len(val_ids)]))

        self.N = N
        self.static_varis = static_varis
        self.splits = {
            'train': np.array([ts_id_to_ind[i] for i in train_ids], dtype=np.int64),
            'val':   np.array([ts_id_to_ind[i] for i in val_ids],   dtype=np.int64),
        }
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)

        # Static demographics: same code path as supervised, just no labels.
        data = self.get_static_data(data)

        # Normalise temporal values per variable using train-only statistics.
        means_stds = (
            data.loc[data.ts_id.isin(train_ids)].groupby('variable')
                .agg({'value': ['mean', 'std']})
        )
        means_stds.columns = [c[1] for c in means_stds.columns]
        means_stds.loc[means_stds['std'] == 0, 'std'] = 1.0
        data = data.merge(means_stds.reset_index(), on='variable', how='left')
        data['value'] = (data['value'] - data['mean']) / data['std']

        variables = data.variable.unique()
        var_to_ind = {v: i for i, v in enumerate(variables)}
        V = len(variables)
        args.V = V
        args.logger.write('# TS variables (pretrain): '+str(V))

        # Persist variables + normalisation so finetune can rebuild the same
        # vocabulary and apply the same z-score (matches the official repo's
        # pt_saved_variables.pkl protocol).
        os.makedirs(args.output_dir, exist_ok=True)
        pickle.dump(
            [list(variables), means_stds, float(self.max_minute)],
            open(os.path.join(args.output_dir, 'pt_saved_variables.pkl'), 'wb'),
        )

        # Build per-patient (value, time, variable_idx) arrays. Sorted by
        # minute so we can later slice an observation window by index.
        data = data.sort_values(by=['ts_ind', 'minute'])
        values = [[] for _ in range(N)]
        times  = [[] for _ in range(N)]
        varis  = [[] for _ in range(N)]
        for row in data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(var_to_ind[row.variable])
        self.values, self.times, self.varis = values, times, varis

        # Pre-compute the set of legal anchor timestamps per patient
        # (unique observed minutes >= min_anchor_minutes and < max_minute, with
        # at least one observation in the forecast window after them).
        max_forecast_anchor = self.max_minute - 1e-6  # need room for >1 sample after t_anchor
        self.timestamps = []
        for arr in self.times:
            if not arr:
                self.timestamps.append(np.array([], dtype=float))
                continue
            uniq = np.unique(np.asarray(arr, dtype=float))
            uniq = uniq[(uniq >= self.min_anchor_minutes) & (uniq < max_forecast_anchor)]
            # Drop the last unique timestamp: no observations strictly after it.
            self.timestamps.append(uniq[:-1] if len(uniq) > 1 else np.array([], dtype=float))

        # Patients without any legal anchor are dropped from the cycler.
        drop = {i for i in range(N) if len(self.timestamps[i]) == 0}
        self.splits = {k: np.array([j for j in v if j not in drop], dtype=np.int64)
                       for k, v in self.splits.items()}
        self.train_cycler = CycleIndex(self.splits['train'], args.train_batch_size)
        self.V = V

    @staticmethod
    def _resolve_pretrain_path(args) -> str:
        """Resolve which pickle to load for pretraining.

        Honours an explicit `args.pretrain_dataset` if provided; otherwise
        appends `_pretrain` to `args.dataset` (the supervised default lives
        in `*_finetune.pkl`). Falls back to the legacy single-pickle name if
        a dedicated pretrain pickle isn't on disk yet, so old setups don't
        suddenly break.
        """
        name = getattr(args, 'pretrain_dataset', None) or (args.dataset + '_pretrain')
        candidate = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', name + '.pkl')
        )
        if os.path.exists(candidate):
            return candidate
        legacy = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', args.dataset + '.pkl')
        )
        return legacy

    def get_batch(self, ind=None):
        if ind is None:
            ind = self.train_cycler.get_batch_ind()
        bsz = len(ind)
        V = self.V
        input_values = []
        input_times  = []
        input_varis  = []
        forecast_values = torch.zeros((bsz, V))
        forecast_mask   = torch.zeros((bsz, V), dtype=torch.int)

        for b, i in enumerate(ind):
            anchors = self.timestamps[i]
            t1 = float(np.random.choice(anchors))
            curr_times  = self.times[i]
            curr_values = self.values[i]
            curr_varis  = self.varis[i]

            # t1_ix = first index strictly after t1 (start of forecast window)
            t1_ix = len(curr_times)
            for ix in range(len(curr_times) - 1, -1, -1):
                if curr_times[ix] <= t1:
                    t1_ix = ix + 1
                    break

            # observation window: last `max_obs` triplets in
            # (t1 - obs_window_minutes, t1].
            t0_ix = max(0, t1_ix - self.max_obs)
            min_obs_time = t1 - self.obs_window_minutes
            while t0_ix < t1_ix and curr_times[t0_ix] < min_obs_time:
                t0_ix += 1

            # Shift the window so t1 sits at the supervised input boundary
            # (`obs_window_minutes`). This keeps the relative-time scaling
            # consistent across anchor choices.
            shift = t1 - self.obs_window_minutes
            obs_t = [t - shift for t in curr_times[t0_ix:t1_ix]]
            input_values.append(list(curr_values[t0_ix:t1_ix]))
            input_times.append(obs_t)
            input_varis.append(list(curr_varis[t0_ix:t1_ix]))

            # forecast window: (t1, t1 + forecast_window_minutes]. Take the
            # most-recent observation of each variable seen there.
            t2 = t1 + self.forecast_window_minutes
            t2_ix = t1_ix
            while t2_ix < len(curr_times) and curr_times[t2_ix] <= t2:
                t2_ix += 1
            for ix in range(t2_ix - 1, t1_ix - 1, -1):
                v = curr_varis[ix]
                if forecast_mask[b, v] == 0:
                    forecast_mask[b, v] = 1
                    forecast_values[b, v] = curr_values[ix]

        num_obs = list(map(len, input_values))
        max_obs = max(num_obs) if num_obs else 0
        pad_lens = (max_obs - np.array(num_obs)) if num_obs else np.zeros(bsz, dtype=int)
        values_p = [x + [0.0] * int(l) for x, l in zip(input_values, pad_lens)]
        times_p  = [x + [0.0] * int(l) for x, l in zip(input_times,  pad_lens)]
        varis_p  = [x + [0]   * int(l) for x, l in zip(input_varis,  pad_lens)]
        values_t = torch.FloatTensor(values_p) if max_obs else torch.zeros((bsz, 0))
        times_t  = torch.FloatTensor(times_p)  if max_obs else torch.zeros((bsz, 0))
        varis_t  = torch.IntTensor(varis_p)    if max_obs else torch.zeros((bsz, 0), dtype=torch.int)
        # Time is mapped into [-1, 1] using `obs_window_minutes` as the scale,
        # matching the supervised dataset's `minute/max_minute*2-1` convention.
        if max_obs:
            times_t = times_t / self.obs_window_minutes * 2.0 - 1.0
        obs_mask = torch.IntTensor([[1] * l1 + [0] * int(l2) for l1, l2 in zip(num_obs, pad_lens)]) \
                    if max_obs else torch.zeros((bsz, 0), dtype=torch.int)
        return {
            'values': values_t,
            'times':  times_t,
            'varis':  varis_t,
            'obs_mask': obs_mask,
            'demo': torch.FloatTensor(self.demo[ind]),
            'forecast_values': forecast_values,
            'forecast_mask':   forecast_mask,
        }

