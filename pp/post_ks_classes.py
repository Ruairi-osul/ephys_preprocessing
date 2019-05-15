import numpy as np


class SpikeSortedRecording:
    '''
    Initialised with path to 

    methods:
        Get

    '''

    def __init__(self, path, nchans=32):
        self.path = path
        self.nchans = nchans
        self.raw_data = self.load_data()

    def load_data(self):
        tmp = np.memmap(self.path)
        shape = int(len(tmp) / self.nchans)
        return np.memmap(self.path, dtype=np.int,
                         shape=(shape, self.nchans))

    def get_good_clusters(self):


class DBInserter:
    '''
    Given a SpikeSortedRecording and an engine, can add to DB 
    '''
    pass
