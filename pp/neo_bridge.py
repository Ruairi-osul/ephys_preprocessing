from neo import SpikeTrain
from quantities import s
import pandas as pd
import numpy as np


def df_to_neo(df, stop='max', grouping_col='cluster_id', fs=30000):
    '''convert a spiketime dataframe to a list of neo spike trains
    returns:
        ids, neo_list'''

    if stop == 'max':
        stop = np.max(df['spike_times']) / fs
    g = df.groupby(grouping_col)['spike_times']
    return g.apply(len).index.values, g.apply(_neo_transformer, stop=stop, fs=fs)


def _neo_transformer(col, stop, fs):
    'from a column of spiketimes, create a spiketrain'
    col = col.divide(fs)
    return SpikeTrain(col.values, t_stop=stop, units=s)


def neo_to_df(a_sig_list, ids):
    '''given a list of neo analog signals, returns those signals in a dataframe'''
    df_list = [pd.DataFrame(a_sig) for a_sig in a_sig_list]
    df = pd.concat(df_list, axis=1)
    df.columns = ids
    return df
