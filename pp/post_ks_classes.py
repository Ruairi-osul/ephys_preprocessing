import numpy as np
import pandas as pd
from preprocess import _extract_waveforms, loadContinuous
from functools import partial
from neo_bridge import df_to_neo, neo_to_df
from quantities import s
from elephant.statistics import instantaneous_rate as ifr
from utils import distance_to_smaller_ref
import warnings
from pprint import pprint as pp
import pdb
from pre_ks_classes import ContinuousRecording
from pathlib import Path
from collections import namedtuple
import feather


class SpikeSortedRecording:
    '''

    '''

    def __init__(self, path: Path, extracted: Path, blocks: list, continuous_dirs: dict = None,
                 nchans: int = 32, fs: int = 30000, max_intertrial_interval=2, recording_params=None,
                 verbose=True):

        self.verbose = verbose

        self.path = Path(path) if not isinstance(path, Path) else path
        self.extracted = Path(extracted) if not isinstance(
            extracted, Path) else extracted
        self.extracted = self.extracted.joinpath(path.name)
        self.continuous_dirs = continuous_dirs
        self.blocks = blocks
        self.recording_params = recording_params

        self.nchans = nchans
        self.fs = fs

        self.good_clusters = self.get_cluster_group_ids(group='good')
        self.mua_clusters = self.get_cluster_group_ids(group='mua')
        self.spike_times = self.load_spike_times()
        self.spike_clusters = self.load_spike_clusters()
        self.good_spike_times = self.get_cluster_spiketimes(self.good_clusters)
        self.mua_spike_times = self.get_cluster_spiketimes(self.mua_clusters)
        self.raw_data = self.load_raw_data()

    def load_raw_data(self):
        path = self.path.joinpath(self.path.name + '.dat')
        tmp = np.memmap(path, mode='r', dtype=np.int16)
        shape = int(len(tmp) / self.nchans)
        return np.memmap(path, dtype=np.int16,
                         shape=(shape, self.nchans), mode='r')

    def get_cluster_group_ids(self, group):
        'get set of cluster ids belonging to group'
        cluster_groups = pd.read_csv(
            self.path.joinpath('cluster_groups.csv'), sep='\t')
        vals = cluster_groups.loc[cluster_groups['group'] == group, :]
        return vals['cluster_id'].values

    def load_spike_times(self):
        return np.load(self.path.joinpath('spike_times.npy'))

    def load_spike_clusters(self):
        return np.load(self.path.joinpath('spike_clusters.npy'))

    def get_cluster_spiketimes(self, clusters):
        'create df of spiketimes (sample) and the cluster to which the spike originated'
        df = pd.DataFrame({'cluster_id': self.spike_clusters.flatten(),
                           'spike_times': self.spike_times.flatten()})
        return df.loc[df['cluster_id'].isin(clusters), :]

    def set_waveforms(self):
        if self.verbose:
            print('Extracting waveforms')
        f1 = partial(_extract_waveforms, raw_data=self.raw_data, ret='data')
        f2 = partial(_extract_waveforms, raw_data=self.raw_data, ret='')
        waveforms = self.good_spike_times.groupby(
            'cluster_id')['spike_times'].apply(f1, raw_data=self.raw_data).apply(pd.Series).reset_index()
        chans = self.good_spike_times.groupby('cluster_id')['spike_times'].apply(
            f2, raw_data=self.raw_data).apply(pd.Series).reset_index()
        chans.columns = ['cluster_id', 'channel']
        waveforms.columns = ['cluster_id', 'sample', 'value']
        self.waveforms, self.chans = waveforms, chans

    def set_ifr(self, resolution=None):
        if self.verbose:
            print('Calculating instantaneous firing rates of all neurons')
        warnings.filterwarnings("ignore")
        if resolution is None:
            resolution = s
        ids, st_list = df_to_neo(self.good_spike_times, stop=(
            np.ceil(len(self.raw_data)/self.fs)))

        a_sigs = list(
            map(partial(ifr, sampling_period=resolution), st_list))
        ifr_df = neo_to_df(a_sigs, ids)
        ifr_df = ifr_df.rename_axis('time').reset_index()
        self.ifr = pd.melt(ifr_df, id_vars='time',
                           var_name='cluster_id', value_name='firing_rate')

    def set_discrete_events(self, discrete_event_chan, threshold=2, num_skip=4):
        if self.verbose:
            print('Finding discrete events')
        data = []
        for block_num, block_name in enumerate(self.blocks):
            paths = self.continuous_dirs[block_name]
            if paths is None:
                if block_num == 0:
                    raise IOError('Could not find baseline continuous dir'
                                  f'for {self.path.name}')
                continue
            paths = [paths] if not isinstance(paths, list) \
                else paths
            for path in paths:
                continuous_rec = ContinuousRecording(
                    path, verbose=self.verbose)
                try:
                    continuous_rec.set_single_file(ch=discrete_event_chan)
                    data.append(continuous_rec.data)
                except ValueError:
                    print(f'Data corrupt: {path}')
        if data:
            data = np.concatenate(data)
            data = np.diff(data)
            discrete_events = np.argwhere(data > threshold).flatten()
            self.discrete_events = discrete_events[num_skip:]

    def get_trials_set_lacencies(self, trial_start=0.5, max_intertrial_interval=2):
        '''update good spike_times to include time relivate to last discrete event'''
        try:
            start_times = self.discrete_events - (trial_start * self.fs)
        except AttributeError:
            return
        self.trials = pd.DataFrame({'trial_number': list(range(len(start_times))),
                                    'eshock_onset': self.discrete_events,
                                    'trial_onset': start_times})

        for name, df in zip(['good_spike_times', 'mua_spike_times'], [self.good_spike_times, self.mua_spike_times]):
            if len(df) == 0:
                continue
            g = df.groupby('cluster_id')['spike_times']
            df['trial'] = g.transform(self._get_trial,
                                      trial_starts=self.trials['trial_onset'].values,
                                      trial_numbers=self.trials['trial_number'].values)

            df = df.merge(right=self.trials,
                          left_on='trial',
                          right_on='trial_number',
                          how='left', copy=False)

            df['latency'] = df.spike_times.subtract(df.eshock_onset)

            df = df.drop(['eshock_onset', 'trial_number', 'trial_onset'],
                         axis=1).pipe(
                self._remove_intershock_trials, recording_params=self.recording_params
            )
            setattr(self, name, df)

    @staticmethod
    def _remove_intershock_trials(df, recording_params):
        '''set np.nan as value of trial for spikes occuring after the baseline
        shock but before the second shock period'''
        # TODO make this work for non hamilton-specific recordings
        if 'chal0' not in recording_params:
            return df
        baseshock_end = recording_params['chal0']
        chalshock_start = recording_params['chalshock0']
        df.loc[(df['spike_times'] > baseshock_end) & (
            df['spike_times'] < chalshock_start), 'trial'] = np.nan
        return df

    @staticmethod
    def _get_trial(arroi, trial_starts, trial_numbers):
        '''find closest trial. For each spike, only search
        the set of trials occuring before that spike'''
        idx = np.searchsorted(trial_starts, arroi)
        return pd.Series([int(trial_numbers[i-1]) if 0 < i < len(trial_numbers) else np.nan for i in idx])

    def set_analog_chan(self):
        # TODO for temperature
        pass

    def save(self):
        if self.verbose:
            print(f"Saving data to: {str(self.extracted)}")
        feather.write_dataframe(self.waveforms,  self.extracted.joinpath('waveforms.feather')) \
            if hasattr(self, 'waveforms') else None
        feather.write_dataframe(self.chans,  self.extracted.joinpath('chans.feather')) \
            if hasattr(self, 'chans') else None
        feather.write_dataframe(self.good_spike_times, self.extracted.joinpath('good_spike_times.feather')) \
            if hasattr(self, 'good_spike_times') else None
        feather.write_dataframe(self.mua_spike_times, self.extracted.joinpath('mua_spike_times.feather')) \
            if hasattr(self, 'mua_spike_times') else None
        feather.write_dataframe(self.trials, self.extracted.joinpath('trials.feather')) \
            if hasattr(self, 'trials') else None
        feather.write_dataframe(self.ifr, self.extracted.joinpath('ifr.feather')) \
            if hasattr(self, 'ifr') else None
