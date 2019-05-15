# ephys_preprocessing

This is used to preprocess ephys data. Scripts are very much designed for a specific data format.

Data are collected usng an open ephys box as .continuous files. Many data iles are concatenated and converted to flat binary .dat files before underoing spike sorting using kilosort. Features from spike sorted neurons (e.g. waveforms and spike times) are extracted and inserted into a relational database. 
