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
import numpy as np
import datetime


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
        self.eng = sa.create_engine(constr, pool_size=40, max_overflow=40)
        self.metadata = sa.MetaData()
        for k, v in kwargs.items():
            setattr(self, k, v)

        if 'verbose' not in kwargs:
            self.verbose = True

        self.experiments_table = sa.Table(
            'experiments', self.metadata, autoload=True, autoload_with=self.eng)
        self.experimental_groups_table = sa.Table(
            'experimental_groups', self.metadata, autoload=True, autoload_with=self.eng)

    def add_experiment(self):
        '''Tables to update:
                experiments
                experimental_groups
                experimental_blocks
        '''
        if not (hasattr(self, 'experiment_name') and hasattr(self, 'experimental_groups')
                and hasattr(self, 'experimental_blocks') and hasattr(self, 'probe_dat_dir')):
            raise ValueError(
                'Required data not availible for initialising experiment')

        if self.verbose:
            print(f'Adding experiment: {self.experiment_name}')

        experimental_blocks_table = sa.Table(
            'experimental_blocks', self.metadata, autoload=True, autoload_with=self.eng)

        experiments_data = self._format_experiments()
        res = self.add_table(engine=self.eng, table=self.experiments_table,
                             data=experiments_data, on_duplicate=self.on_duplicate,
                             identidier_col=self.experiments_table.c.experiment_name,
                             identifier_data=experiments_data['experiment_name'])
        self.experiment_id = res.inserted_primary_key[0]

        experimental_groups_data = self._format_experimental_groups()
        experimental_blocks_data = self._format_experimental_blocks()
        _ = self.add_table(engine=self.eng, table=self.experimental_groups_table,
                           data=experimental_groups_data)
        _ = self.add_table(engine=self.eng, table=experimental_blocks_table,
                           data=experimental_blocks_data)

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
        if self.verbose:
            print(f'adding recording: {self.recordings_params["name"]}')

        recordings_table = sa.Table(
            'recordings', self.metadata, autoload=True, autoload_with=self.eng)
        neurons_table = sa.Table(
            'neurons', self.metadata, autoload=True, autoload_with=self.eng)
        good_spike_times_table = sa.Table(
            'good_spike_times', self.metadata, autoload=True, autoload_with=self.eng)
        waveforms_table = sa.Table(
            'waveform_timepoints', self.metadata, autoload=True, autoload_with=self.eng)
        multi_units_table = sa.Table(
            'multi_units', self.metadata, autoload=True, autoload_with=self.eng)
        mua_spike_times_table = sa.Table(
            'mua_spike_times', self.metadata, autoload=True, autoload_with=self.eng)
        ifr_table = sa.Table(
            'ifr', self.metadata, autoload=True, autoload_with=self.eng)
        eshock_table = sa.Table(
            'eshock_events', self.metadata, autoload=True, autoload_with=self.eng)

        try:
            recordings_data = self._format_recordings()
            res = self.add_table(engine=self.eng, table=recordings_table,
                                 data=recordings_data,
                                 on_duplicate=self.on_duplicate, identidier_col=recordings_table.c.recording_name,
                                 identifier_data=recordings_data['recording_name'])
            self.recording_id = res.inserted_primary_key[0]
        except AssertionError:
            raise ValueError(f'Insufficient preprocessing for insertion for'
                             f'recording: \t{self.recordings_params["name"]}')

        try:
            neurons_data = self._format_neurons()
            assert neurons_data
            self.add_table(engine=self.eng, table=neurons_table, data=neurons_data,
                           on_duplicate=self.on_duplicate)

            good_spiketimes_data = self._format_good_spiketimes(neurons_table)
            assert good_spiketimes_data
            self.add_table(engine=self.eng, table=good_spike_times_table, data=good_spiketimes_data,
                           on_duplicate=self.on_duplicate)
        except AssertionError:
            print('Single unit data not availible for recording: '
                  f'\t{self.recordings_params["name"]}')
            pass

        try:
            waveforms_data = self._format_waveforms(neurons_table)
            self.add_table(engine=self.eng, table=waveforms_table, data=waveforms_data,
                           on_duplicate=self.on_duplicate)
        except AssertionError:
            print(
                f'No waveform data availible: recording'
                f'\t{self.recordings_params["name"]}')
            pass

        try:
            mua_data = self._format_multiunits()
            assert mua_data
            self.add_table(engine=self.eng, table=multi_units_table,
                           data=mua_data, on_duplicate=self.on_duplicate)

            mua_spike_times = self._format_mua_spiketimes(multi_units_table)
            self.add_table(engine=self.eng, table=mua_spike_times_table,
                           data=mua_spike_times, on_duplicate=self.on_duplicate)
        except AssertionError:
            print('No multi unit data availible: recording'
                  f'\t{self.recordings_params["name"]}')
            pass

        try:
            ifr_data = self._format_ifr(neurons_table)
            assert ifr_data
            self.add_table(engine=self.eng, table=ifr_table,
                           data=ifr_data, on_duplicate=self.on_duplicate)
        except AssertionError:
            print('No ifr data availible: recording'
                  f'\t{self.recordings_params["name"]}')
            pass

        try:

            eshock_data = self._format_eshock_events()
            assert eshock_data
            self.add_table(engine=self.eng, table=eshock_table,
                           data=eshock_data, on_duplicate=self.on_duplicate)
        except AssertionError:
            print('No trial data availible: recording'
                  f'\t{self.recordings_params["name"]}')
            pass

    @staticmethod
    def get_ids(engine, new_data, ref_table,
                ref_table_contraint_col, ref_table_contraint_data,
                merge_col, id_col):
        with engine.connect() as con:
            ref_data = con.execute(sa.select([ref_table]
                                             ).where(
                                                 ref_table_contraint_col == ref_table_contraint_data)).fetchall()
        ref_data = pd.DataFrame(ref_data, columns=ref_data[0].keys())[
            [id_col, merge_col]]
        res = pd.merge(new_data, ref_data, on=merge_col)
        assert len(res) == len(new_data), 'error finding ids'
        return res

    @staticmethod
    def add_table(engine, table, data, on_duplicate=None,
                  identidier_col=None, identifier_data=None):
        '''identifier col is the column to use to check if the data has already been added
        identifier data is the value in data used to identify whether data is already present in the
        db'''
        with engine.connect() as con:
            try:
                res = con.execute(insert(table, data))
            except IntegrityError:
                if (on_duplicate == 'fail') or (on_duplicate == 'skip'):
                    raise ValueError(
                        f'Dubplicate entry found in {on_duplicate} mode when inserting')
                elif on_duplicate == 'redo':
                    assert (identidier_col is not None) and (
                        identifier_data is not None)
                    con.execute(sa.delete(table).where(
                        identidier_col == identifier_data))
                res = con.execute(insert(table), data)
        return res

    @staticmethod
    def to_dict(df):
        return list(df.to_dict('index').values())

    @staticmethod
    def nan_to_null(df):
        return df.where(pd.notnull(df), None)

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
        recording['recording_date'] = datetime.datetime.strptime(
            self.recordings_params['date'], '%Y-%m-%d').date()
        recording['start_time'] = datetime.datetime.strptime(
            self.recordings_params['start_time'], '%H-%M-%S').time()
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
        assert hasattr(self, 'chans') and hasattr(self, 'recording_id')
        return pd.DataFrame(data={'cluster_id': self.chans['cluster_id'].values,
                                  'max_amp_channel': self.chans['channel'].values,
                                  'excluded': np.nan,
                                  'recording_id': self.recording_id}
                            ).pipe(
                                self.nan_to_null).pipe(
            self.to_dict
        )

    def _format_good_spiketimes(self, neurons_table):
        assert hasattr(self, 'good_spike_times')
        return self.get_ids(engine=self.eng, new_data=self.good_spike_times,
                            ref_table=neurons_table,
                            ref_table_contraint_col=neurons_table.c.recording_id,
                            ref_table_contraint_data=self.recording_id,
                            merge_col='cluster_id', id_col='neuron_id'
                            ).pipe(
                                lambda x: x.drop('cluster_id', axis=1)
        ).drop_duplicates().pipe(
            self.nan_to_null
        ).pipe(self.to_dict)

    def _format_waveforms(self, neurons_table):
        assert hasattr(self, 'waveforms')
        return self.get_ids(engine=self.eng, new_data=self.waveforms,
                            ref_table=neurons_table,
                            ref_table_contraint_col=neurons_table.c.recording_id,
                            ref_table_contraint_data=self.recording_id,
                            merge_col='cluster_id', id_col='neuron_id'
                            ).pipe(
                                lambda x: x.drop('cluster_id', axis=1)
        ).pipe(
            self.nan_to_null
        ).pipe(self.to_dict)

    def _format_multiunits(self):
        assert hasattr(self, 'recording_id') and hasattr(
            self, 'mua_spike_times')
        return pd.DataFrame(data={'cluster_id': self.mua_spike_times['cluster_id'].unique(),
                                  'recording_id': self.recording_id}
                            ).pipe(self.to_dict)

    def _format_mua_spiketimes(self, mua_table):
        assert hasattr(self, 'recording_id') and hasattr(
            self, 'mua_spike_times')
        return self.get_ids(engine=self.eng, new_data=self.mua_spike_times,
                            ref_table=mua_table,
                            ref_table_contraint_col=mua_table.c.recording_id,
                            ref_table_contraint_data=self.recording_id,
                            merge_col='cluster_id', id_col='mua_id'
                            ).pipe(
                                lambda x: x.drop('cluster_id', axis=1)
        ).drop_duplicates().pipe(
            self.nan_to_null
        ).pipe(self.to_dict)

    def _format_ifr(self, neurons_table):
        assert hasattr(self, 'ifr')
        data = self.ifr.rename(axis='columns', mapper={
                               'time': 'timepoint_sec'})
        return self.get_ids(engine=self.eng, new_data=data,
                            ref_table=neurons_table,
                            ref_table_contraint_col=neurons_table.c.recording_id,
                            ref_table_contraint_data=self.recording_id,
                            merge_col='cluster_id', id_col='neuron_id'
                            ).pipe(
                                lambda x: x.drop('cluster_id', axis=1)
        ).pipe(
            self.nan_to_null
        ).pipe(self.to_dict)

    def _format_eshock_events(self):
        assert hasattr(self, 'recording_id')
        assert hasattr(self, 'recording_id')
        assert hasattr(self, 'trials')
        data = self.trials
        data['recording_id'] = self.recording_id
        return data.pipe(self.nan_to_null).pipe(self.to_dict)

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
                           'trials.feather', 'waveforms.feather', 'chans.feather']

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

        except FileNotFoundError:
            print(f'Error during file import: {str(file)}')
            pass
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
