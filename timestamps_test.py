
# %%
from sqlalchemy import create_engine, select
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


engine = create_engine('mysql+pymysql://ruairi:@localhost/eshock')


# %%
def relative(spikes_vec, ref):
    idx = np.searchsorted(ref, spikes_vec, side='right')
    return spikes_vec-ref[idx-1]


# %%
cut_off = 30000 * 60 * 60
name = 'ESHOCK_04_LOC1'


# %%
q2 = f'''
SELECT eshock_events.event_sample FROM eshock_events 
JOIN recordings ON eshock_events.recording_id=recordings.recording_id
WHERE recordings.dat_filename='{name}'
'''


# %%
events = pd.read_sql(q2, engine)
e_vec = events['event_sample'].values
max_t = max(e_vec)
min_t = min(e_vec)


# %%
q = f'''
SELECT neurons.neuron_id, spike_times.spike_times FROM spike_times 
JOIN neurons ON spike_times.neuron_id=neurons.neuron_id 
JOIN recordings ON neurons.recording_id=recordings.recording_id
WHERE (spike_times >= {min_t}) & (spike_times <= {max_t} ) & (recordings.dat_filename='{name}')
'''


# %%
spikes = pd.read_sql(q, engine)


# %%
neurons = spikes['neuron_id'].unique()


# %%
for neuron in neurons:
    spikes_neuron = spikes.loc[spikes['neuron_id'] == neuron]
    s_vec = spikes_neuron['spike_times'].values
    rel = relative(spikes_vec=s_vec, ref=e_vec)
    rel_t = rel / 300
    plt.hist(rel_t[(rel_t > 0) & (rel_t < 200)], bins=100)
    plt.xlabel('Time Sinse Last Electric Show [Ms]')
    plt.title(f'neuron_id {neuron}')
    plt.ylabel('Probability of Spiking')
    plt.show()
