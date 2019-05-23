from pre_ks_classes import PreKilosortPreprocessor
import json
import argparse
from pathlib import Path
import datetime


def _get_options():
    parser = argparse.ArgumentParser(
        description='Create .dat files for kilosort spike sorting')
    parser.add_argument('-f', '--file_mapper', required=True,
                        help='path to .json file with mappings')
    parser.add_argument('-c', '--config', required=True,
                        help='path to experiment config file')
    parser.add_argument('-m', '--mode', default='skip',
                        help='action to take if')
    parser.add_argument('-l', '--log_mode', default='w',
                        help='mode for writing to the logfile. specify "a" if NOT deleting and writing anew')
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        out = json.load(f)
    return out


def save_json(recordings_mapper, path):
    with open(path, 'w') as f:
        json.dump(recordings_mapper, f)


def check_log(name, logfile):
    with open(logfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        assert name not in line.split(',')


def get_pp_options(experiment_settings):
    '''get nessary experiment settings'''
    continuous_home = Path(experiment_settings['directories']['continuous_dir']
                           ) if 'continuous_dir' in experiment_settings['directories'] else None
    tmp_home = Path(experiment_settings['directories']['tmp_dat_dir']
                    ) if 'tmp_dat_dir' in experiment_settings['directories'] else None
    probe_dat_dir = Path(experiment_settings['directories']['probe_dat_dir']
                         ) if 'probe_dat_dir' in experiment_settings['directories'] else None
    extracted = Path(experiment_settings['directories']['extracted']
                     ) if 'extracted' in experiment_settings['directories'] else None
    log_file = experiment_settings['log_files']['pre_kilosort']
    chan_map = experiment_settings['recording_config']['probe_chanmap']
    blocks = experiment_settings['recording_config']['blocks']
    return continuous_home, tmp_home, probe_dat_dir, extracted, log_file, chan_map, blocks


def make_continuous_dirs_abs(continuous_home, continuous_dirs):
    dirs = {}
    for block, path in continuous_dirs.items():
        if isinstance(path, list):
            dirs[block] = list(
                map(lambda x: continuous_home.joinpath(x), path))
        else:
            dirs[block] = continuous_home.joinpath(
                path) if path is not None else None
    return dirs


if __name__ == "__main__":
    args = vars(_get_options())
    experiment_settings = load_json(args['config'])
    recordings_mapper = load_json(args['file_mapper'])

    continuous_home, tmp_home, probe_dat_dir, extracted, log_file, chan_map, blocks = get_pp_options(
        experiment_settings)

    for ind, recording in enumerate(recordings_mapper.values()):
        if (recording['todo'] != 'yes') or ('continuous_dirs' not in recording):
            print('skipping\n{}\n'.format(recording['name']))
            continue
        try:
            check_log(recording['name'], log_file)
        except AssertionError:
            if args['mode'] == 'fail':
                raise ValueError('')
            elif args['mode'] == 'skip':
                continue

        continuous_dirs = make_continuous_dirs_abs(
            continuous_home, recording['continuous_dirs'])

        processor = PreKilosortPreprocessor(
            name=recording['name'],
            continuous_files=continuous_dirs,
            extracted=extracted,
            date=recording['date'],
            chan_map=chan_map,
            blocks=blocks,
            tmp_dir=tmp_home,
            dat_dir=probe_dat_dir,
            group_id=recording['group_id'])
        processor.create_dat()
        processor.create_recordings_params()

        logmode = args['log_mode'] if ind == 0 else 'a'
        with open(log_file, logmode) as f:
            line = ','.join(
                [recording['name'], str(datetime.datetime.now()), '\n'])
            f.write(line)

        recording['todo'] = 'done'
        save_json(recordings_mapper, args['file_mapper'])
