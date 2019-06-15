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

pre = params['pre0_samples']
baseshock = params['baseshock0_samples']
post_baseshock = params['post_baseshock0_samples']
chal = params['chal0_samples']
chalshock = params['chalshock0_samples']
post_chalshock = params['post_chalshock0_samples']
way = params['way0_samples']

times = np.array([pre, baseshock, post_baseshock, chal, chalshock,
                  post_chalshock, way])

pdb.set_trace()
