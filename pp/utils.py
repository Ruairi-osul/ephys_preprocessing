import numpy as np
import pandas as pd
import scipy.io
import time
import os
from glob import glob

# constants
NUM_HEADER_BYTES = 1024
SAMPLES_PER_RECORD = 1024
BYTES_PER_SAMPLE = 2
RECORD_SIZE = 4 + 8 + SAMPLES_PER_RECORD * BYTES_PER_SAMPLE + \
    10  # size of each continuous record in bytes
RECORD_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255])

# constants for pre-allocating matrices:
MAX_NUMBER_OF_SPIKES = int(1e6)
MAX_NUMBER_OF_RECORDS = int(1e6)
MAX_NUMBER_OF_EVENTS = int(1e6)


def distance_to_smaller_ref(arroi, ref):
    '''Given an array of interest and a reference array, 
    find the difference between each element of the array of interest
    to the corresponding element in reference array which is smaller and minimises the distance between the reference element and the array of interest element'''
    idx = np.searchsorted(ref, arroi, side='right')
    return arroi-ref[idx-1]


def gen_spikes_ts_df(spike_clusters, spike_times, good_cluster_nums):
    data = {'cluster_id': spike_clusters.flatten(),
            'spike_times': spike_times.flatten()}
    df = pd.DataFrame(data)
    df = df.loc[df['cluster_id'].isin(good_cluster_nums), :]
    return df


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
            raise ValueError(
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


def get_good_cluster_numbers(cluster_groups_df, mua=False):
    '''
    Takes the cluster_groups pandas DataFrame fomed during data loading and returns a numpy array of cluster
    ids defined as 'Good' during kilosort and phy spike sorting
    Parameters:
        cluster_groups_df   = the pandas DataFrame containing information on which cluster is 'Good', 'Noise' etc.
    Returns:
        A numpy array of 'Good' cluster ids
    '''
    if mua:
        good_clusters_df = cluster_groups_df.loc[(
            cluster_groups_df['group'] == 'good') | (cluster_groups_df['group'] == 'mua'), :]
    else:
        good_clusters_df = cluster_groups_df.loc[cluster_groups_df['group'] == 'good', :]
    return good_clusters_df['cluster_id'].values


def loadFolderToArray(folderpath, channels='all', chprefix='CH',
                      dtype=float, session='0', source='100'):
    '''Load continuous files in specified folder to a single numpy array. By default all
    CH continous files are loaded in numerical order, ordering can be specified with
    optional channels argument which should be a list of channel numbers.'''

    if channels == 'all':
        channels = _get_sorted_channels(folderpath, chprefix, session, source)

    if session == '0':
        filelist = [source + '_' + chprefix + x +
                    '.continuous' for x in map(str, channels)]
    else:
        filelist = [source + '_' + chprefix + x + '_' +
                    session + '.continuous' for x in map(str, channels)]

    t0 = time.time()
    numFiles = 1

    channel_1_data = loadContinuous(os.path.join(
        folderpath, filelist[0]), dtype)['data']

    n_samples = len(channel_1_data)
    n_channels = len(filelist)

    data_array = np.zeros([n_samples, n_channels], dtype)
    data_array[:, 0] = channel_1_data

    for i, f in enumerate(filelist[1:]):
        data_array[:, i +
                   1] = loadContinuous(os.path.join(folderpath, f), dtype)['data']
        numFiles += 1

    print(''.join(('Avg. Load Time: ', str((time.time() - t0) / numFiles), ' sec')))
    print(''.join(('Total Load Time: ', str((time.time() - t0)), ' sec')))

    return data_array


def downsample(trace, down):
    downsampled = scipy.signal.resample(trace, np.shape(trace)[0] / down)
    return downsampled


def _get_sorted_channels(folderpath, chprefix='CH', session='0', source='100'):
    Files = [f for f in os.listdir(folderpath) if '.continuous' in f
             and '_' + chprefix in f
             and source in f]

    if session == '0':
        Files = [f for f in Files if len(f.split('_')) == 2]
        Chs = sorted([int(f.split('_' + chprefix)[1].split('.')[0])
                      for f in Files])
    else:
        Files = [f for f in Files if len(f.split('_')) == 3
                 and f.split('.')[0].split('_')[2] == session]

        Chs = sorted([int(f.split('_' + chprefix)[1].split('_')[0])
                      for f in Files])

    return(Chs)


def readHeader(f):
    header = {}
    h = f.read(1024).decode().replace('\n', '').replace('header.', '')
    for i, item in enumerate(h.split(';')):
        if '=' in item:
            header[item.split(' = ')[0]] = item.split(' = ')[1]
    return header


def _walklevel(some_dir, level=1):
    '''generator similar to os.walk but with optional
    argument for the number of levels to search down
    from parent
    '''
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def _has_ext(path, ext):
    if '.' not in ext:
        ext = ''.join(['.', ext])
    return bool(glob(os.path.join(path, ''.join(['*', ext]))))


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))
