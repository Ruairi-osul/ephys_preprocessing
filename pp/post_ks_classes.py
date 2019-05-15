import numpy as np


class SpikeSortedRecording:
    '''
    Initialised with path to 

    methods:
        Get

    '''

    def __init__(self, path, extracted, nchans=32):
        self.path = path
        self.nchans = nchans
        self.extracted = extracted
        self.raw_data = self.load_data()

    def load_data(self):
        tmp = np.memmap(self.path)
        shape = int(len(tmp) / self.nchans)
        return np.memmap(self.path, dtype=np.int,
                         shape=(shape, self.nchans))

    def get_good_clusters(self):
        pass

    def get_waveforms(self):
        pass


class DBInserter:
    '''
    Given a SpikeSortedRecording and an engine, can add to DB 
    '''
    pass
