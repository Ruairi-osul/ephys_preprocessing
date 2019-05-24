import numpy as np
import pandas as pd
from preprocess import _extract_waveforms
from functools import partial
from pprint import pprint as pp
import pdb


class SpikeSortedRecording:
    '''
    path should be a pathlib object
    Usage:
        processor = SpikeSortedRecording(path, extracted)
        processir.extract_spiketimes()
        processor.extract_waveforms()
        processor.calculate_ifr()
        processor.get_trail_sampl

    methods:
        Get

    '''

    def __init__(self, path, extracted, nchans=32):
        self.path = path
        self.nchans = nchans
        self.extracted = extracted
        self.good_clusters = self.get_cluster_group_ids(group='good')
        self.mua_clusters = self.get_cluster_group_ids(group='mua')

        self.spike_times = self.load_spike_times()
        self.spike_clusters = self.load_spike_clusters()
        self.good_spike_times = self.get_cluster_spiketimes(self.good_clusters)
        self.mua_spike_times = self.get_cluster_spiketimes(self.mua_clusters)

        self.raw_data = self.load_raw_data()

    def load_raw_data(self):
        path = self.path.joinpath(self.path.name + '.dat')
        tmp = np.memmap(path)
        shape = int(len(tmp) / self.nchans)
        return np.memmap(path, dtype=np.int,
                         shape=(shape, self.nchans))

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
        f1 = partial(_extract_waveforms, raw_data=self.raw_data, ret='data')
        f2 = partial(_extract_waveforms, raw_data=self.raw_data, ret='')
        waveforms = self.good_spike_times.groupby(
            'cluster_id')['spike_times'].apply(f1, raw_data=self.raw_data).apply(pd.Series).reset_index()
        chans = self.good_spike_times.groupby('cluster_id')['spike_times'].apply(
            f2, raw_data=self.raw_data).apply(pd.Series).reset_index()
        chans.columns = ['cluster_id', 'channel']
        waveforms.columns = ['cluster_id', 'sample', 'value']
        return waveforms, chans


class DBInserter:
    '''
    Given a SpikeSortedRecording and an engine, can add to DB 
    '''
    pass


if __name__ == '__main__':
    from pathlib import Path

    recording = Path('/media/ruairi/big_bck/CITWAY/probe_dat_dir/acute_05')
    processor = SpikeSortedRecording(recording, extracted='')
    pdb.set_trace()
