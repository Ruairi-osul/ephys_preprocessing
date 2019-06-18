import numpy as np
from scipy import signal


class Analog_signal:

    def __init__(self):
        pass

    @property
    def time(self):
        assert hasattr(self, 'fs')
        assert hasattr(self, 'data')
        return np.linspace(1, len(self.data), len(self.data)) / self.fs

    def downsample(self, new_fs):
        assert hasattr(self, fs)
        pass


def main():
    # get continuous mapper and log files

    # get set of completed recordings

    # loop over recordings

        # if recording is already done, decide what to do

        # loop over EEG chans
            # load the data as a continuous signal
            # instantiate the analog signal
            # downsample
            # save to extracted
            # update log file

        # loop over LFP chans
            # load the data as a continuous signal
            # instantiate the analog signal
            # downsample
            # save to extracted
            # update log file
