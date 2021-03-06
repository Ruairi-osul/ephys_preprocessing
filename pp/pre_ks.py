from pre_ks_classes import PreKilosortPreprocessor
import json
import argparse
from pathlib import Path
import datetime
import pdb


def _get_options():
    parser = argparse.ArgumentParser(
        description='All meta and probe-related preprocessing that can'
                    'be done pre kilosort')

    parser.add_argument('-c', '--config', required=True,
                        help='Path to experiment config file')
    parser.add_argument('-m', '--mode', default='full',
                        help='Set behaviour between {"full", "params"}'
                        'Specify "params" if getting recordings params only.'
                        'Defaults to "full".')
    parser.add_argument('-d', '--on_duplicate', default='skip',
                        help='Action to take when duplicates are found.'
                        'Set behaviour between {"skip", "fail", "redo"}, already completed recordings')
    parser.add_argument('-l', '--log_mode', default='a',
                        help='mode for writing to the logfile. specify "w" if'
                        'deleting and writing anew. Defaults to "a"')

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
    recordings_mapper = load_json(
        experiment_settings['directories']['continuous_mapper'])

    continuous_home, tmp_home, probe_dat_dir, extracted, log_file, chan_map, blocks = get_pp_options(
        experiment_settings)

    for ind, recording in enumerate(recordings_mapper.values()):

        # Does this recording need to be done?
        if ((recording['todo'] != 'yes') and (args['on_duplicate'] != 'redo') or ('continuous_dirs' not in recording)):
            print('skipping\n{}\n'.format(recording['name']))
            continue
        # Has this recording been done?
        try:
            check_log(recording['name'], log_file)
        except AssertionError:
            if args['on_duplicate'] == 'fail':
                raise ValueError('duplicate found')
            elif args['on_duplicate'] == 'skip':
                continue
            elif args['on_duplicate'] == 'redo':
                pass

        continuous_dirs = make_continuous_dirs_abs(
            continuous_home, recording['continuous_dirs'])

        processor = PreKilosortPreprocessor(
            experiment_name=experiment_settings['exp_name'],
            name=recording['name'],
            continuous_files=continuous_dirs,
            extracted=extracted,
            date=recording['date'],
            chan_map=chan_map,
            blocks=blocks,
            tmp_dir=tmp_home,
            dat_dir=probe_dat_dir,
            group_id=recording['group_id'])

        if args['mode'] == 'params':                # for just params
            processor.get_blocklengths()
            processor.create_recordings_params()
        elif args['mode'] == 'full':                # for creating dat
            processor.create_dat()
            processor.create_recordings_params()
        else:
            raise ValueError(
                'Unknown argument for "--mode" parameter.'
                'Should "full" or "skip"')

        logmode = args['log_mode'] if ind == 0 else 'a'
        with open(log_file, logmode) as f:
            line = ','.join(
                [recording['name'], str(datetime.datetime.now()), '\n'])
            f.write(line)

        recording['todo'] = 'done'
        save_json(recordings_mapper,
                  experiment_settings['directories']['continuous_mapper'])
