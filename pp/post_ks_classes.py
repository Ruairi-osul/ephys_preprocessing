import numpy as np
import pandas as pd


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
        self.raw_data = self.load_raw_data()
        self.good_clusters = self.get_cluster_group_ids(group='good')
        self.mua_clusters = self.get_cluster_group_ids(group='mua')
        self.spike_times = self.load_spike_times()
        self.load_spike_clusters = load_spike_clusters

    def load_raw_data(self):
        tmp = np.memmap(self.path)
        shape = int(len(tmp) / self.nchans)
        return np.memmap(self.path, dtype=np.int,
                         shape=(shape, self.nchans))

    def get_cluster_group_ids(self, group):
        cluster_groups = pd.read_csv(
            self.path.joinpath('cluster_groups.csv'), sep='\t')
        cluster_groups = cluster_groups.loc[cluster_groups['group'] == group, :]
        return cluster_groups['cluster_id'].values

    def load_spike_times(self):
        return np.load(self.path.joinpath('spike_times.npy'))

    def load_spike_clusters(self):
        return np.load(self.path.joinpath('spike_clusters.npy'))

    def get_cluster_spiketimes(self, clusters):
        df = pd.DataFrame({'cluster_id': self.spike_clusters.flatten(),
                           'spike_times': self.spike_times.flatten()})
        return df.loc[df['cluster_id'].isin(clusters), :]

    def get_waveforms(self):
        pass


class DBInserter:
    '''
    Given a SpikeSortedRecording and an engine, can add to DB 
    '''
    pass
