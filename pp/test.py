from utils import loadEvents, loadContinuous
import pandas as pd
from pathlib import Path
import pdb


cont_root = Path('/media/ruairi/big_bck/HAMILTON/continuous')

block_name = 'HAMILTON_34_2019-07-30_12-11-19_CIT_SHOCK'

file = '120_CH14.continuous'
events = 'all_channels.events'

events = loadEvents(cont_root.joinpath(block_name).joinpath(events))

# load continuous file for that block
data = loadContinuous(cont_root.joinpath(block_name).joinpath(file))
first_timestamp = data['timestamps'][0]

df = pd.DataFrame(
    {'channel': events['channel'],
     'timestamps': events['timestamps'],
     'eventid': events['eventId']})
df = df[(df['eventid'] == 1) & (df['channel'] == int('4'))]
df = df.assign(timestamps=lambda x: x['timestamps'] - first_timestamp)\
    .pipe(lambda x: x.iloc[4:, :])
pdb.set_trace()


# get timestamp of the first sample of that block


# subtract the first timestamp from the timestamps col of the df
