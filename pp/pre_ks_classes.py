from utils import loadFolderToArray
import numpy as np
import os
from pathlib import Path
import json
from pprint import pprint as pp
import re
import datetime


class ContinuousRecording:
    '''
    Has block and experiment attributes.

    Methods:
        Load
        Common average reference:
        Save to dat:

    '''

    def __init__(self, path, chan_map, name=None, verbose=False):
        self.path = path  # (n_samples, n_channels)
        self.chan_map = chan_map
        self.verbose = verbose
        self.name = path if name is None else name
        self.data = self.load_data()
        self.block_len = self.get_block_len()

    def load_data(self):
        if self.verbose:
            print(f'Loading data: {self.name}\nPath:{self.path}')
        data = None
        session_options = ['0', '1', '2', '3']
        source_options = ['100', '120']
        working = False
        for source_option in source_options:
            for session_option in session_options:
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
        return data

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
    One per each recording session.
    Attributes:
        chan_map
        recording_name
        continuous_files
        blocks !!! must be in order

    Methods:

        create_dat:
            For each continuous file:
                load
                common average reference
                create tmp .dat file
            concat tmp dat files

        create_recordings_params:
            load block_lengths.json
            create recording params.json: {date, block_lengths, start_time, group_id}

    Output:
        recording_name.dat 
        recordings_params.json (for insertion into the recordings table)
    '''

    def __init__(self, name, continuous_files, chan_map, date,
                 blocks, group_id, extracted=None, tmp_dir=None, dat_dir=None, verbose=True):
        'continuous_files should be a dictionary'

        self.verbose = verbose
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

    def _get_start_time(self):
        first_block_name = self.blocks[0]
        first_block_file = str(self.continuous_files[first_block_name])
        time_pattern = '_\d{2}-\d{2}-\d{2}_'
        start_idx = re.search(time_pattern, first_block_file).start()
        end_idx = re.search(time_pattern, first_block_file).end()
        return first_block_file[start_idx + 1: end_idx - 1]

    def create_recordings_params(self):
        params_out = {"name": self.name,
                      "group_id": self.group_id, "date": self.date}
        params_out.update(self.blocklenghts)
        fname = self.extracted.joinpath('recordings_params.json')
        with fname.open('w') as f:
            json.dump(params_out, f, index=2)


class TimestampExtractor:

    def __init__(self, path, extracted, threshold=2.5):
        self.path = path
        self.extracted = extracted

    def load_data(self):
        pass

    def get_timestamps(self):
        pass
