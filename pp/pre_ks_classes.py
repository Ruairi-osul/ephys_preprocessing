from utils import loadFolderToArray, loadContinuous
import numpy as np
import os
from pathlib import Path
import json
from pprint import pprint as pp
import re
import datetime
import pdb


class ContinuousRecording:
    ''' Class for handling data stored as .continuous files
    Initialise with a path to the containing folder

    Methods:
        set_probe_data: use chan_map attribute to load in probe data
        set_single_file: load a single file. Can specify alternative to the default channel 1
        common_average_reference: apply a common average reference to all channels stored in the data attr
        get_block_len: get the number of samples in the recording
        save_datfile: save the contents of the data attr to a .dat file
    '''
    session_options = ['0', '1', '2', '3']
    source_options = ['100', '120']

    def __init__(self, path, chan_map=None, name=None, verbose=False):
        if isinstance(path, str):
            from pathlib import Path
            path = Path(path)
        self.path = path  # (n_samples, n_channels)
        self.chan_map = chan_map
        self.verbose = verbose
        self.name = path if name is None else name

    def set_probe_data(self):
        if self.verbose:
            print(f'Loading data: {self.name}\nPath:{self.path}')
        if self.chan_map is None:
            raise ValueError('Trying to load probe data without chan map')
        data = None
        working = False
        for source_option in self.source_options:
            for session_option in self.session_options:
                try:
                    data = loadFolderToArray(self.path, channels=self.chan_map, dtype=np.int16,
                                             chprefix='CH', session=session_option, source=source_option)
                    working = True
                    break
                except:
                    continue
            if working:
                break

        if data is None:
            raise IOError(
                'Could not load continuous files\n{}'.format(str(self.path)))
        self.data = data

    def set_single_file(self, ch='1'):
        'specify ch as a string integer'
        data = None
        working = False
        for source_option in self.source_options:
            for session_option in self.session_options:
                if session_option == '0':
                    p = source_option + '_' + 'CH' + ch + '.continuous'
                else:
                    p = source_option + '_' + 'CH' + ch + '_' + session_option + '.continuous'
                try:
                    # pdb.set_trace()
                    data = loadContinuous(str(self.path.joinpath(p)))['data']
                    working = True
                    break
                except:
                    continue
            if working:
                break

        if data is None:
            raise IOError(
                'Could not load continuous files\n{}'.format(str(self.path)))
        self.data = data

    def common_average_reference(self):
        if self.verbose:
            print('Referenceing all channels')
        reference = np.mean(self.data, 1)
        for i in range(self.data.shape[1]):
            self.data[:, i] = self.data[:, i] - reference

    def get_block_len(self):
        return self.data.shape[0]

    def save_datfile(self, fileout):
        if self.verbose:
            print(f'Saving data: {fileout}')
        self.data.tofile(fileout)


class PreKilosortPreprocessor:
    '''
    class for handling prekilosort preprocessing of continuous files
    methods:
        _create_dirs: creates directories if necessary for output. Does not overwrite if exists.
        create_dat: uses chan_map to identify probe channels, 
                    concatenates data across blocks and applies a common average reference to each
                    saves results as a .dat file which can be used during kilosort spike sorting.
                    saves the blocklengths of each recording
        get_block_lengths:  used when creating .dat files is not necessary. 
                            loads in a single .continuous file per block and stores the blocklengths
        create_recording_params: 
    '''

    def __init__(self, name, continuous_files, chan_map,
                 blocks, date=None, group_id=None, experiment_name=None, extracted=None, tmp_dir=None, dat_dir=None, verbose=True):
        'continuous_files should be a dictionary'

        self.verbose = verbose
        self.experiment_name = experiment_name
        self.blocks = blocks
        self.chan_map = chan_map
        self.group_id = group_id
        self.name = name
        self.date = date
        self.continuous_files = continuous_files
        tmp_dir, dat_dir, extracted = self._create_dirs(
            tmp_dir, dat_dir, extracted)
        self.extracted = extracted
        self.tmp_dir = tmp_dir
        self.dat_dir = dat_dir
        self.out_file = dat_dir.joinpath(self.name + '.dat')
        self.blocklenghts = {}
        self.tmp_files = []
        self.start_time = self._get_start_time()

    def _create_dirs(self, tmp_dir, dat_dir, extracted):
        if (tmp_dir is None) or (dat_dir is None) or (extracted is None):
            cwd = Path(os.getcwd())
            if tmp_dir is None:
                tmp_dir = cwd.joinpath('tmp_datfiles')
            if dat_dir is None:
                dat_dir = cwd.joinpath('probe_datfiles')
            if extracted is None:
                extracted = cwd.joinpath('extracted')

        tmp_dir = tmp_dir.joinpath(self.name)
        dat_dir = dat_dir.joinpath(self.name)
        extracted = extracted.joinpath(self.name)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        dat_dir.mkdir(exist_ok=True, parents=True)
        extracted.mkdir(exist_ok=True, parents=True)
        return tmp_dir, dat_dir, extracted

    def create_dat(self):
        if self.verbose:
            print(f'{self.name}: Creating tmp dat files')

        for block_num, block_name in enumerate(self.blocks):
            paths = self.continuous_files[block_name]
            if paths is None:
                if block_num == 0:
                    raise IOError(
                        f'Error preprocessing {self.name}\nCould not find {block_name}')
                self.blocklenghts[block_name + '0_samples'] = None
                continue
            if not isinstance(paths, list):
                paths = [paths]
            for i, path in enumerate(paths):
                continous_rec = ContinuousRecording(
                    path, chan_map=self.chan_map, name=block_name, verbose=self.verbose)
                continous_rec.set_probe_data()
                continous_rec.common_average_reference()
                file_out = self.tmp_dir.joinpath(
                    '_'.join([self.name, block_name, str(i)])+'.dat')
                continous_rec.save_datfile(str(file_out))
                self.blocklenghts[block_name +
                                  f'{str(i)}_samples'] = continous_rec.get_block_len()
                self.tmp_files.append(file_out)  # should be in order

        if self.verbose:
            print(f'Concatenating files: {str(self.out_file)}')
        if len(self.tmp_files) == 1:
            cmd = f'mv {self.tmp_files[0]} {str(self.out_file)}'
            os.system(cmd)
        else:
            files = ' '.join(list(map(str, self.tmp_files)))
            cmd = f'cat {files} > {str(self.out_file)}'
            os.system(cmd)
            for tmp in self.tmp_files:
                os.remove(str(tmp))

    def get_blocklengths(self):
        if self.verbose:
            print(f'{self.name}: Gettng block_lengths')

        for block_num, block_name in enumerate(self.blocks):
            paths = self.continuous_files[block_name]
            if paths is None:
                if block_num == 0:
                    raise IOError(
                        f'Error preprocessing {self.name}\nCould not find {block_name}')
                self.blocklenghts[block_name + '0_samples'] = None
                continue
            if not isinstance(paths, list):
                paths = [paths]
            for i, path in enumerate(paths):
                continous_rec = ContinuousRecording(
                    path, verbose=self.verbose)
                continous_rec.set_single_file()
                self.blocklenghts[block_name +
                                  f'{str(i)}_samples'] = continous_rec.get_block_len()

    def _get_start_time(self):
        first_block_name = self.blocks[0]
        first_block_file = str(self.continuous_files[first_block_name])
        time_pattern = r'_\d{2}-\d{2}-\d{2}_'
        start_idx = re.search(time_pattern, first_block_file).start()
        end_idx = re.search(time_pattern, first_block_file).end()
        return first_block_file[start_idx + 1: end_idx - 1]

    def create_recordings_params(self):
        # TODO add start time
        params_out = {"name": self.name,
                      "exp_name": self.experiment_name,
                      "group_id": self.group_id,
                      "date": self.date,
                      "start_time": self.start_time}
        params_out.update(self.blocklenghts)
        fname = self.extracted.joinpath('recordings_params.json')
        with fname.open('w') as f:
            json.dump(params_out, f, indent=2)
