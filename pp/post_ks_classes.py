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


class SpikeSortedRecording:
    '''
    path should be a pathlib object

    methods:
        Get

    '''

    def __init__(self, path, extracted, adc=None, nchans=32, resolution=None, fs=30000, max_intertrial_interval=2, verbose=True):
        self.path = path
        self.verbose = verbose
        self.nchans = nchans
        self.extracted = extracted
        self.fs = fs
        self.good_clusters = self.get_cluster_group_ids(group='good')
        self.mua_clusters = self.get_cluster_group_ids(group='mua')

        self.spike_times = self.load_spike_times()
        self.spike_clusters = self.load_spike_clusters()
        self.good_spike_times = self.get_cluster_spiketimes(self.good_clusters)
        self.mua_spike_times = self.get_cluster_spiketimes(self.mua_clusters)

        self.raw_data = self.load_raw_data()
        #self.waveforms, self.chans = self.get_waveforms()
        self.ifr = self.get_ifr(resolution=resolution, fs=self.fs)

        if adc is not None:
            self.get_timestamps(adc)

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

    def get_waveforms(self):
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
        return waveforms, chans

    def get_ifr(self, fs=30000, resolution=None):
        if self.verbose:
            print('Calculating instantaneous firing rates of all neurons')
        if resolution is None:
            resolution = s
        ids, st_list = df_to_neo(self.good_spike_times, stop=(
            np.ceil(len(self.raw_data)/fs)))
        warnings.filterwarnings("ignore")
        a_sigs = list(
            map(partial(ifr, sampling_period=resolution), st_list))
        ifr_df = neo_to_df(a_sigs, ids)
        ifr_df = ifr_df.rename_axis('time').reset_index()
        return pd.melt(ifr_df, id_vars='time', var_name='cluster_id', value_name='firing_rate')

    def get_timestamps(self, adc):
        pdb.set_trace()
        # TODO: ADD continuous path to be adc
        #continuous = loadContinuous(self.path.joinpath(adc))

    def get_trials_set_lacencies(self, trial_start=0.5, max_intertrial_interval=2):
        # g = self.good_spike_times.groupby('cluster_id')['spike_times]
        # distance = g.transform(distance_to_smaller_ref, ref=self.timestamps)
        # latency = distance - int(trial_start * self.fs)
        # max_latency = int(max_intertrial_interval * self.fs)
        # latency[np.where( (latency > max_latency) | (latency < int(trial_start * self.fs)) )] = np.nan
        pass

    def save(self):
        pass


class DBInserter:
    '''
    Given a SpikeSortedRecording and an engine, can add to DB
    '''
    pass


if __name__ == '__main__':
    from pathlib import Path

    recording = Path('/media/ruairi/big_bck/CITWAY/probe_dat_dir/acute_02')
    processor = SpikeSortedRecording(
        recording, extracted='', adc='120_CH59.continuous')
