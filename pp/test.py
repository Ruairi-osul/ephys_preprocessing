import feather
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

p = '/media/ruairi/big_bck/HAMILTON/extracted/hamilton_09/good_spike_times.feather'
df = feather.read_dataframe(p)

p = '/media/ruairi/big_bck/HAMILTON/extracted/hamilton_09/recordings_params.json'
with open(p) as f:
    params = json.load(f)

baseshock_start = params['baseshock0']
baseshock_end = params['chal0']
chalshock_start = params['chalshock0']
chalshock_end = params['post_chalshock0']

trials = feather.read_dataframe(
    '/media/ruairi/big_bck/HAMILTON/extracted/hamilton_09/trials.feather')

df.loc[(df['spike_times'] > baseshock_end) & (
    df['spike_times'] < chalshock_start), 'trial'] = np.nan
pdb.set_trace()
