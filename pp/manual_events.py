import numpy as np
from pathlib import Path

name = 'hamilton_23'

baseshock_start = 1814.13
baseshock_stop = 2394.13

chalshock_start = 4231.621
chalshock_stop = 4801.621

# probably dont change these
FS = 30000
INTER_SHOCK_INTERVAL = 2
outname = 'manual_events.npy'
extracted_path = '/media/ruairi/big_bck/HAMILTON/extracted'
out_path = Path(extracted_path).joinpath(name).joinpath(outname)

# creating the evenets
if (baseshock_start is not None) and (baseshock_stop is not None):
    baseshock_events = np.arange(
        baseshock_start, baseshock_stop, INTER_SHOCK_INTERVAL) * FS

if (chalshock_start is not None) and (chalshock_stop is not None):
    chalshock_events = np.arange(
        chalshock_start, chalshock_stop, INTER_SHOCK_INTERVAL) * FS
else:
    chalshock_events = None

if chalshock_events is not None:
    events = np.concatenate(
        [baseshock_events, chalshock_events]).astype(int)
else:
    events = baseshock_events

np.save(str(out_path), events)
