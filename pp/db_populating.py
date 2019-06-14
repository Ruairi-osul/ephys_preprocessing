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
    # TODO add mode for deleted and reinserting

    def __init__(self, constr, **kwargs):
        self.eng = sa.create_engine(constr)
        self.metadata = sa.MetaData()
        for k, v in kwargs.items():
            setattr(self, k, v)

        if 'verbose' not in kwargs:
            self.verbose = True

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

        experiments_table = sa.Table(
            'experiments', self.metadata, auto_load=True, autoload_with=self.eng)
        experimental_groups_table = sa.Table(
            'experimental_groups', self.metadata, autoload=True, autoload_with=self.eng)
        experimental_blocks_table = sa.Table(
            'experimental_blocks', self.metadata, autoload=True, autoload_with=self.eng)

        experiments_data = self._format_experiments()
        with self.eng.connect() as con:
            try:

                res = con.execute(insert(experiments_table), experiments_data)
            except IntegrityError:
                if self.on_duplicate == 'fail':
                    raise ValueError(
                        'Dubplicate entry found in fail mode when inserting experimental data')
                elif self.on_duplicate == 'skip':
                    raise ValueError(
                        f'Error in logging {self.experiment_name} marked as todo but a db entry already exists')
                elif self.on_duplicate == 'redo':
                    con.execute(sa.delete(experiments_table).where(
                        experiments_table.c.experiment_name == experiments_data['experiment_name']))
                    res = con.execute(
                        insert(experiments_table), experiments_data)

            self.experiment_id = res.inserted_primary_key[0]

        experimental_groups_data = self._format_experimental_groups()
        experimental_blocks_data = self._format_experimental_blocks()
        with self.eng.connect() as con:
            con.execute(insert(experimental_groups_table),
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
        return pd.DataFrame(self.experimental_blocks, index=[1]).melt(id_vars='experiment_id', var_name='block_name', value_name='block_len').pipe(
            self.to_dict
        )

    def _format_recordings(self):
        pass

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
                           'mua_spike_times.feather', 'recording_params.json',
                           'trials.feather', 'waveforms.feather']

    extracted_files = list(
        map(lambda x: str(root.joinpath(x)), extracted_files))

    extracted_data = {}
    for file in extracted_files:
        try:
            if file.split('.')[-1] == '.feather':
                extracted_data[file.split('.')[0]] = feather.read_dataframe(f)
            elif file.split('.')[-1] == '.json':
                with open(file) as f:
                    extracted_data[file.split('.')[0]] = json.load(f)

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

    pdb.set_trace()
    # get done recordings
    log_in_path = config['log_files']['postkilosort']
    log_out_path = config['log_files']['db_insersion']

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
            db_str, on_duplicate=on_duplicate, **extracted_data)
        inserter.add_recording()


if __name__ == "__main__":
    db_str = 'mysql+pymysql://ruairi:@localhost/ephys'
    args = _get_options()
    main(db_str, args.config, args.on_duplicate, args.init)
