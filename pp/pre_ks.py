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
    return parser.parse_args()


def main(recording, chan_map, blocks, tmp_home, dat_dir, group_id, date, extracted):
    processor = PreKilosortPreprocessor(
        name=recording['name'],
        continuous_files=dirs,
        extracted=extracted,
        date=recording['date'],
        chan_map=chan_map,
        blocks=blocks,
        tmp_dir=tmp_home,
        dat_dir=probe_dat_dir,
        group_id=recording['group_id'])
    processor.create_dat()
    processor.create_recordings_params()

    with open(log_file, 'a') as f:
        line = ','.join(
            [recording['name'], str(datetime.datetime.now()), '\n'])
        f.write(line)


def load_json(path):
    with open(path) as f:
        out = json.load(f)
    return out


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def check_log(name, logfile):
    with open(logfile, 'r') as f:
        lines = f.readlines()
    for line in lines:
        assert name not in line.split(',')


if __name__ == "__main__":
    args = vars(_get_options())
    ops = load_json(args['config'])
    data = load_json(args['file_mapper'])

    continuous_home = Path(ops['directories']['continuous_dir']
                           ) if 'continuous_dir' in ops['directories'] else None
    tmp_home = Path(ops['directories']['tmp_dat_dir']
                    ) if 'tmp_dat_dir' in ops['directories'] else None
    probe_dat_dir = Path(ops['directories']['probe_dat_dir']
                         ) if 'probe_dat_dir' in ops['directories'] else None
    extracted = Path(ops['directories']['extracted']
                     ) if 'extracted' in ops['directories'] else None

    log_file = ops['log_files']['pre_kilosort']
    chan_map = ops['recording_config']['probe_chanmap']
    blocks = ops['recording_config']['blocks']

    if len(data) == 1:
        data = [data]

    for _, recording in data.items():

        if recording['todo'] != 'yes':
            continue
        try:
            check_log(recording['name'], log_file)
        except AssertionError:
            if args['mode'] == 'fail':
                raise ValueError('')
            elif args['mode'] == 'skip':
                continue
        if 'continuous_dirs' not in recording:
            print(
                'skipping\n{}\n'.format(recording['name']))
            break
        dirs = {}
        for block, path in recording['continuous_dirs'].items():
            if isinstance(path, list):
                dirs[block] = list(
                    map(lambda x: continuous_home.joinpath(x), path))
            else:
                dirs[block] = continuous_home.joinpath(
                    path) if path is not None else None

        main(recording, chan_map, blocks, tmp_home,
             dat_dir=probe_dat_dir, group_id=recording['group_id'],
             date=recording['date'], extracted=extracted)
        recording['todo'] = 'done'

    save_json(data, args['file_mapper'])
