import argparse
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import insert
import feather
import json
import pandas as pd
import csv
from pathlib import Path
import pdb
from sqlalchemy.exc import IntegrityError


def _get_options():
    parser = argparse.ArgumentParser(
        description='Add data to the ephys mysql database')

    parser.add_argument('-i', '--init', required=False,
                        action='store_true',
                        help='Use this flag if initialising the database. Otherwise leave blank')
    parser.add_argument('-d', '--on_duplicate', required=False,
                        default="fail",
                        help='Action to take when finding duplicate entries.'
                        'Choose between {fail, skip, update}'
                        'Defaults to fail')
    parser.add_argument('-c', '--config', required=True,
                        help='Path to config .json file')
    return parser.parse_args()


class DBInserter:

    def __init__(self, constr, **kwargs):
        self.eng = sa.create_engine(constr)
        self.metadata = sa.MetaData()
        for k, v in kwargs.items():
            setattr(self, k, v)

        if 'verbose' not in kwargs:
            self.verbose = True

        self.experiments_table = sa.Table(
            'experiments', self.metadata, auto_load=True, autoload_with=self.eng)
        self.experimental_groups_table = sa.Table(
            'experimental_groups', self.metadata, autoload=True, autoload_with=self.eng)

    def add_experiment(self):
        '''Tables to update:
                experiments
                experimental_groups
                experimental_blocks
        '''
        if not (hasattr(self, 'experiment_name') and hasattr(self, 'experimental_groups') and hasattr(self, 'experimental_blocks') and hasattr(self, 'probe_dat_dir')):
            raise ValueError(
                'Required data not availible for initialising experiment')

        if self.verbose:
            print(f'Adding experiment: {self.experiment_name}')

        experimental_blocks_table = sa.Table(
            'experimental_blocks', self.metadata, autoload=True, autoload_with=self.eng)

        experiments_data = self._format_experiments()
        with self.eng.connect() as con:
            try:

                res = con.execute(
                    insert(self.experiments_table), experiments_data)
            except IntegrityError:
                if self.on_duplicate == 'fail':
                    raise ValueError(
                        'Dubplicate entry found in fail mode when inserting experimental data')
                elif self.on_duplicate == 'skip':
                    raise ValueError(
                        f'Error in logging {self.experiment_name} marked as todo but a db entry already exists')
                elif self.on_duplicate == 'redo':
                    con.execute(sa.delete(self.experiments_table).where(
                        self.experiments_table.c.experiment_name == experiments_data['experiment_name']))
                    res = con.execute(
                        insert(self.experiments_table), experiments_data)

            self.experiment_id = res.inserted_primary_key[0]

        experimental_groups_data = self._format_experimental_groups()
        experimental_blocks_data = self._format_experimental_blocks()
        with self.eng.connect() as con:
            con.execute(insert(self.experimental_groups_table),
                        experimental_groups_data)
            con.execute(insert(experimental_blocks_table),
                        experimental_blocks_data)

    def add_recording(self):
        '''Tables to update:
                recordings
                block_lengths
                neurons
                waveforms
                spiketimes
                temperature
                eeg
                lfp
        '''
        # loop over options

        # check if
        recordings_table = sa.Table(
            'recordings', self.metadata, auto_load=True, autoload_with=self.eng)
        recordings_data = self._format_recordings()
        with self.eng.connect() as con:
            try:
                res = con.execute(insert(recordings_table, recordings_data))
            except IntegrityError:
                if self.on_duplicate == 'fail':
                    raise ValueError(
                        'Dubplicate entry found in fail mode when inserting recording data')
                elif self.on_duplicate == 'skip':
                    raise ValueError(
                        f'Error in logging {self.recordings_params['name']} marked as todo but a db entry already exists')
                elif self.on_duplicate == 'redo':
                    con.execute(sa.delete(recordings_table).where(
                        self.recordings_table.c.recording_name == recordings_data['recording_name']))
                    res = con.execute(
                        insert(recordings_table), recordings_data)

            self.recording_id = res.inserted_primary_key[0]
        pass

        pass

    @staticmethod
    def to_dict(df):
        return list(df.to_dict('index').values())

    def _format_experiments(self):
        return {'experiment_name': self.experiment_name,
                'probe_dat_dir': self.probe_dat_dir}

    def _format_experimental_groups(self):
        assert hasattr(self, 'experiment_id')
        self.experimental_groups['experiment_id'] = self.experiment_id
        return self.experimental_groups.drop(
            'experiment_name', axis=1).rename(
            {'group_id': 'group_code'}, axis=1).pipe(
            self.to_dict)

    def _format_experimental_blocks(self):
        assert hasattr(self, 'experiment_id')
        self.experimental_blocks['experiment_id'] = self.experiment_id
        return pd.DataFrame(
            self.experimental_blocks, index=[1]
        ).melt(
            id_vars='experiment_id', var_name='block_name',
            value_name='block_len'
        ).pipe(
            self.to_dict
        )

    def _format_recordings(self):
        # TODO add checks to see if values have been defined
        assert self.recordings_params
        recording = {}
        recording['recording_name'] = self.recordings_params['name']
        recording['recording_date'] = self.recordings_params['date']
        recording['start_time'] = self.recordings_params['start_time']
        recording['eeg_fs'] = self.eeg_fs
        recording['probe_fs'] = self.probe_fs
        recording['dat_filename'] = str(Path(self.probe_dat_dir
                                             ).joinpath(self.recordings_params['name']
                                                        ).joinpath(''.join([self.recordings_params['name'],
                                                                            '.dat']
                                                                           )))
        stmt = sa.select([self.experimental_groups_table.c.id]
                         ).select_from(
                             self.experimental_groups_table.join(
                                 self.experiments_table,
                                 self.experiments_table.c.experiment_id == self.experimental_groups_table.c.experiment_id
                             )).where(
                                 sa.and_(
                                     self.experimental_groups_table.c.group_code == self.recordings_params[
                                         'group_id'],
                                     self.experiments_table.c.experiment_name == self.recordings_params[
                                         'exp_name']
                                 ))

        with self.eng.connect() as con:
            recording['group_id'] = con.execute(stmt).fetchone()[0]
        return recording

    def _format_neurons(self):
        pass

    def _format_spiketimes(self):
        pass

    def _format_waveforms(self):
        pass

    def _format_temperature(self):
        pass

    def _format_eeg(self):
        pass

    def _format_lfp(self):
        pass


def load_experimental_data(config):
    experimental_data = {}
    experimental_data['experiment_name'] = config['exp_name']
    experimental_data['probe_dat_dir'] = config['directories']['probe_dat_dir']
    experimental_data['experimental_groups'] = pd.read_csv(
        config['directories']['experimental_groups'])
    experimental_data['experimental_blocks'] = config['recording_config']['block_lenghts']
    return experimental_data


def load_extracted_data(root: Path, extracted_files=None):
    if extracted_files is None:
        extracted_files = ['good_spike_times.feather', 'ifr.feather',
                           'mua_spike_times.feather', 'recordings_params.json',
                           'trials.feather', 'waveforms.feather']

    extracted_files = list(
        map(lambda x: root.joinpath(x), extracted_files))

    extracted_data = {}
    for file in extracted_files:
        try:
            if file.suffix == '.feather':
                extracted_data[file.name.split(
                    '.')[0]] = feather.read_dataframe(file)
            elif file.suffix == '.json':
                with file.open() as f:
                    extracted_data[file.name.split('.')[0]] = json.load(f)

        except FileNotFoundError as e:
            print('Error during file import')
            raise e
    return extracted_data


def main(db_str, config, on_duplicate, init=False):

    with open(config) as f:
        config = json.load(f)

    if init:
        experimental_data = load_experimental_data(config)
        inserter = DBInserter(
            db_str, on_duplicate=on_duplicate, **experimental_data)
        inserter.add_experiment()

    # get done recordings
    log_in_path = config['log_files']['post_kilosort']
    log_out_path = config['log_files']['db_inserted']

    with open(log_in_path, 'r') as f:
        reader = csv.reader(f)
        kilosorted_recordings = [row[0] for row in reader if row]
    with open(log_out_path, 'r') as f:
        reader = csv.reader(f)
        inserted_recordings = [row[0] for row in reader if row]

    for recording in kilosorted_recordings:
        if recording in inserted_recordings:
            if on_duplicate == 'fail':
                raise ValueError(
                    'Previously-inserted recording attempted to be inserted in "fail" mode')
            elif on_duplicate == 'skip':
                print(f'skipping {recording}')
            elif on_duplicate == 'redo':
                pass
            else:
                raise ValueError('Could not parse on_duplicate behaviour.')

        extracted_data = load_extracted_data(
            Path(config['directories']['extracted']).joinpath(recording))

        inserter = DBInserter(
            db_str, on_duplicate=on_duplicate,
            probe_dat_dir=config['directories']['probe_dat_dir'],
            eeg_fs=config['recording_config']['eeg_fs'],
            probe_fs=config['recording_config']['probe_fs'],
            **extracted_data)
        inserter.add_recording()


if __name__ == "__main__":
    db_str = 'mysql+pymysql://ruairi:@localhost/ephys'
    args = _get_options()
    main(db_str, args.config, args.on_duplicate, args.init)
