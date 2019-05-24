import numpy as np
import pandas as pd
import os
from copy import deepcopy
from functools import partial
from utils import (gen_spikes_ts_df, get_good_cluster_numbers,
                   loadFolderToArray, _get_sorted_channels,
                   readHeader)


def get_spike_times(p, r_id=None, mua=False):
    '''Given a path to a dictory containing kilosort files,
    returns a pandas dataframe with spike times of clusters
    marked as good during spike sorting. You can optionally specify
    a the recording id for further identification'''
    spk_c, spk_tms, c_gps = load_kilosort_arrays(p)
    clusters = get_good_cluster_numbers(c_gps, mua)
    df = gen_spikes_ts_df(spk_c, spk_tms, clusters)
    if r_id is not None:
        df['recording_name'] = r_id
    return df


def load_dat_data(p, n_chans=32):
    tmp = np.memmap(p, dtype=np.int16)
    shp = int(len(tmp) / n_chans)
    return np.memmap(p, dtype=np.int16,
                     shape=(shp, n_chans))


def get_waveforms(spike_data, rd):
    '''Given a pandas df of spike times and the path to
    a the parent directory of the .dat file containing the raw
    data for that recording, extracts waveforms for each cluester
    and the channel on which that cluster had the highest amplitude

    params:
        spike_data: pandas df of spike times and cluster ids as cols
        rid
    '''
    raw_data = load_dat_data(p=os.path.join(
        rd, os.path.basename(rd)) + '.dat')
    f1 = partial(_extract_waveforms, raw_data=raw_data, ret='data')
    f2 = partial(_extract_waveforms, raw_data=raw_data, ret='')

    waveforms = spike_data.groupby('cluster_id')['spike_times'].apply(
        f1, raw_data=raw_data).apply(pd.Series).reset_index()

    chans = spike_data.groupby('cluster_id')[
        'spike_times'].apply(f2,
                             raw_data=raw_data).apply(pd.Series).reset_index()

    chans.columns = ['cluster_id', 'channel']
    waveforms.columns = ['cluster_id', 'sample', 'value']
    return waveforms, chans


def _extract_waveforms(spk_tms, raw_data, ret='data',
                       n_spks=600, n_samps=240, n_chans=32):
    assert len(spk_tms) > n_spks, 'Not ennough spikes'
    spk_tms = spk_tms.values
    window = np.arange(int(-n_samps / 2), int(n_samps / 2))
    wvfrms = np.zeros((n_spks, n_samps, n_chans))
    for i in range(n_spks):
        srt = int(spk_tms[i] + window[0])
        end = int(spk_tms[i] + window[-1] + 1)
        srt = srt if srt > 0 else 0
        try:
            wvfrms[i, :, :] = raw_data[srt:end, :]
        except ValueError:
            filler = np.empty((n_samps, n_chans))
            filler[:] = np.nan
            wvfrms[i, :, :] = filler
    wvfrms = pd.DataFrame(np.nanmean(wvfrms, axis=0),
                          columns=range(1, n_chans + 1))
    norm = wvfrms - np.mean(wvfrms)
    tmp = norm.apply(np.min, axis=0)
    good_chan = tmp.idxmin()
    wvfrms = wvfrms.loc[:, good_chan]
    if ret == 'data':
        return wvfrms
    else:
        return good_chan


def load_kilosort_arrays(parent_dir):
    '''
    Loads arrays generated during kilosort into numpy arrays and pandas DataFrames
    Parameters:
        parent_dir       = name of the parent_dir being analysed
    Returns:
        spike_clusters  = numpy array of len(num_spikes) identifying the cluster from which each spike arrose
        spike_times     = numpy array of len(num_spikes) identifying the time in samples at which each spike occured
        cluster_groups  = pandas DataDrame with one row per cluster and column 'cluster_group' identifying whether
                          that cluster had been marked as 'Noise', 'MUA' or 'Good'
    '''
    try:
        spike_clusters = np.load(os.path.join(
            parent_dir, 'spike_clusters.npy'))
        spike_times = np.load(os.path.join(parent_dir, 'spike_times.npy'))
        cluster_groups = pd.read_csv(os.path.join(
            parent_dir, 'cluster_groups.csv'), sep='\t')
    except IOError:
        print('Error loading Kilosort Files. Files not found')
        raise
    try:  # check data quality
        assert np.shape(spike_times.flatten()) == np.shape(spike_clusters)
    except AssertionError:
        AssertionError('Array lengths do not match in parent_dir {}'.format(
            parent_dir))
    return spike_clusters, spike_times, cluster_groups


def loadContinuous(filepath, dtype=float):

        # constants
    NUM_HEADER_BYTES = 1024
    SAMPLES_PER_RECORD = 1024
    BYTES_PER_SAMPLE = 2
    RECORD_SIZE = 4 + 8 + SAMPLES_PER_RECORD * BYTES_PER_SAMPLE + \
        10  # size of each continuous record in bytes

    assert dtype in (float, np.int16), \
        'Invalid data type specified for loadContinous, valid types are float and np.int16'

    ch = {}

    # read in the data
    f = open(filepath, 'rb')

    fileLength = os.fstat(f.fileno()).st_size

    # calculate number of samples
    recordBytes = fileLength - NUM_HEADER_BYTES
    if recordBytes % RECORD_SIZE != 0:
        raise Exception('''File size is not consistent with a
                        continuous file: may be corrupt''')
    nrec = recordBytes // RECORD_SIZE
    nsamp = nrec * SAMPLES_PER_RECORD
    # pre-allocate samples
    samples = np.zeros(nsamp, dtype)
    timestamps = np.zeros(nrec)
    recordingNumbers = np.zeros(nrec)
    indices = np.arange(0, nsamp + 1, SAMPLES_PER_RECORD, np.dtype(np.int64))

    header = readHeader(f)

    recIndices = np.arange(0, nrec)

    for recordNumber in recIndices:

        timestamps[recordNumber] = np.fromfile(f, np.dtype(
            '<i8'), 1)  # little-endian 64-bit signed integer
        # little-endian 16-bit unsigned integer
        N = np.fromfile(f, np.dtype('<u2'), 1)[0]

        # print index

        if N != SAMPLES_PER_RECORD:
            raise Exception(
                'Found corrupted record in block ' + str(recordNumber))

        # big-endian 16-bit unsigned integer
        recordingNumbers[recordNumber] = (np.fromfile(f, np.dtype('>u2'), 1))

        if dtype == float:  # Convert data to float array and convert bits to voltage.
            # big-endian 16-bit signed integer, multiplied by bitVolts
            data = np.fromfile(f, np.dtype('>i2'), N) * \
                float(header['bitVolts'])
        else:  # Keep data in signed 16 bit integer format.
            # big-endian 16-bit signed integer
            data = np.fromfile(f, np.dtype('>i2'), N)
        samples[indices[recordNumber]:indices[recordNumber + 1]] = data

        marker = f.read(10)  # dump

    # print recordNumber
    # print index

    ch['header'] = header
    ch['timestamps'] = timestamps
    ch['data'] = samples  # OR use downsample(samples,1), to save space
    ch['recordingNumber'] = recordingNumbers
    f.close()
    return ch


def pack_2(folderpath, filename='', channels='all', chprefix='CH',
           dref=None, session='0', source='100'):
    '''numpy array.tofile wrapper. Loads .continuous files in folderpath,
    (channels specidied by channels), applies reference and saves as .dat

    filename: Name of the output file. By default, it follows the same layout of continuous files,
              but without the channel number, for example, '100_CHs_3.dat' or '100_ADCs.dat'.

    channels:  List of channel numbers specifying order in which channels are packed. By default
               all CH continous files are packed in numerical order.

    chprefix:  String name that defines if channels from headstage, auxiliary or ADC inputs
               will be loaded.

    dref:  Digital referencing - either supply a channel number or 'ave' to reference to the
           average of packed channels.

    source: String name of the source that openephys uses as the prefix. It is usually 100,
            if the headstage is the first source added, but can specify something different.

    '''

    data_array = loadFolderToArray(
        folderpath, channels, chprefix, np.int16, session, source)

    if dref:
        if dref == 'ave':
            print('Digital referencing to average of all channels.')
            reference = np.mean(data_array, 1)
        else:
            print('Digital referencing to channel ' + str(dref))
            if channels == 'all':
                channels = _get_sorted_channels(
                    folderpath, chprefix, session, source)
            reference = deepcopy(data_array[:, channels.index(dref)])
        for i in range(data_array.shape[1]):
            data_array[:, i] = data_array[:, i] - reference

    if session == '0':
        session = ''
    else:
        session = '_' + session

    if not filename:
        filename = source + '_' + chprefix + 's' + session + '.dat'
    print('Packing data to file: ' + filename)
    data_array.tofile(os.path.join(folderpath, filename))
