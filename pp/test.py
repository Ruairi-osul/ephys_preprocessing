import feather
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np

p = '/media/ruairi/big_bck/HAMILTON/extracted/hamilton_09/ifr.feather'
ifr = feather.read_dataframe(p)
p = '/media/ruairi/big_bck/HAMILTON/extracted/hamilton_09/chans.feather'
waves = feather.read_dataframe(p)

pdb.set_trace()
