import argparse
import json
import pdb
from pprint import pprint as pp
import csv
from pathlib import Path
from post_ks_classes import SpikeSortedRecording
import datetime


def _get_options():
    parser = argparse.ArgumentParser(
        description='')

    parser.add_argument('-c', '--config_file', required=True,
                        help='path to experiment config file')
    parser.add_argument('-d', '--on_duplicate', default='skip',
                        help='action to take on duplicate\n'
                        'choose one of: {"skip", "fail", "redo"}\n'
                        'defaults to "skip"')
    parser.add_argument('-l', '--log_mode', default='a',
                        help='logging behaviour.\n'
                        'choose betwee {"a", "w"}\n'
                        'defaults to "a"')
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        out = json.load(f)
    return out


def main(config_path, on_duplicate, log_mode):
    # load params
    experiment_params = load_json(config_path)
    continuous_mapper = load_json(
        experiment_params['directories']['continuous_mapper'])
    continuous_dir = Path(experiment_params['directories']['continuous_dir'])

    log_in_path = experiment_params['log_files']['kilosort']
    log_out_path = experiment_params['log_files']['post_kilosort']
    if on_duplicate != 'redo':
        with open(log_out_path, 'r') as f:
            reader = csv.reader(f)
            done = [row[0] for row in reader]
    else:
        done = []

    with open(log_in_path, 'r') as f:
        reader = csv.reader(f)
        kilosorted_recordings = [Path(row[0]) if Path(row[0]).is_absolute() else Path(
            experiment_params['directories']['probe_dat_dir']).joinpath(row[0])
            for row in reader if row]

    for i, recording in enumerate(continuous_mapper.values()):
        # has continuous_mapper been properly completed for this recording?
        if 'continuous_dirs' not in recording:
            print('Continuous dirs field not found.'
                  f'Skipping {recording["name"]}')
        # Has kilosort been done?
        try:
            kilosort_path = next(
                filter(lambda x: x.name == recording['name'], kilosorted_recordings))
        except StopIteration:
            print(f'Kilosort not yet completed: {recording["name"]}\n'
                  'continuing...')
            continue
        # has post_ks already been ran on this recording?
        try:
            assert recording['name'] not in done
        except AssertionError:
            if on_duplicate == 'skip':
                print(f'Duplicate found.'
                      f'Skipping {recording["name"]}')
                continue
            elif on_duplicate == 'fail':
                raise ValueError('Duplicate found in "fail" mode')
            elif on_duplicate == 'redo':
                pass

        continuous_dirs = {block_name: continuous_dir.joinpath(d)
                           if d is not None else None
                           for block_name, d in recording['continuous_dirs'].items()}

        eshock_chan = recording['adc_chans']['eshock_ttl'] \
            if ('adc_chans' in recording) and ('eshock_ttl' in recording['adc_chans']) \
            else None
        temperature_chan = recording['adc_chans']['temperature'] \
            if ('adc_chans' in recording) and ('temperature' in recording['adc_chans']) \
            else None

        processor = SpikeSortedRecording(path=kilosort_path,
                                         extracted=experiment_params['directories']['extracted'],
                                         continuous_dirs=continuous_dirs,
                                         blocks=experiment_params['recording_config']['blocks'])

        if len(processor.good_clusters) > 0:
            processor.set_waveforms()
            processor.set_ifr()

        # pdb.set_trace()
        if eshock_chan is not None:
            processor.set_discrete_events(eshock_chan)
            processor.get_trials_set_lacencies()
        if temperature_chan is not None:
            # TODO!!!
            pass
        processor.save()

        if i == 0 and log_mode == 'w':
            logmode = 'w'
        else:
            logmode = 'a'

        with open(log_out_path, logmode) as f:
            line = ','.join(
                [recording['name'], str(datetime.datetime.now()), '\n'])
            f.write(line)


if __name__ == "__main__":
    args = _get_options()
    main(args.config_file, args.on_duplicate, args.log_mode)
