DROP DATABASE IF EXISTS ephys;
CREATE DATABASE ephys;
USE ephys;


CREATE TABLE experiments(
    experiment_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    probe_dat_dir VARCHAR(300)
);

CREATE TABLE experimental_blocks(
    experiment_id INT NOT NULL,
    block_name VARCHAR(250) NOT NULL,
    block_len INT,
    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE,
    PRIMARY KEY (experiment_id, block_name)
);

CREATE TABLE experimental_groups (
    group_id INT,
    group_name VARCHAR(250),
    experiment_id INT,
    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE
    PRIMARY KEY (group_id, experiment_id)
);

CREATE TABLE mice (
    mouse_id INT AUTO_INCREMENT PRIMARY KEY,
    genotype VARCHAR(100),
    mass DOUBLE,
    sex VARCHAR(100),
    chronic_treatment VARCHAR(200),
    viral_injection VARCHAR(200)
);

CREATE TABLE recordings (
    recording_id INT AUTO_INCREMENT PRIMARY KEY,
    recording_name VARCHAR(250),
    recording_date DATE,
    start_time TIME,
    dat_filename VARCHAR(250) NOT NULL,
    group_id INT,
    excluded INT,
    mouse_id INT,
    probe_fs INT,
    eeg_fs INT,
    FOREIGN KEY (group_id)
        REFERENCES experimental_groups(group_id)
        ON DELETE CASCADE,
    FOREIGN KEY (mouse_id)
        REFERENCES mice(mouse_id)
);

CREATE TABLE recording_block_lenghts (
    recording_id INT NOT NULL,
    block_name INT NOT NULL,
    samples BIGINT,
    FOREIGN KEY (recording_id)
        REFERENCES recordings(recording_id)
        ON DELETE CASCADE,
    PRIMARY KEY (recording_id, block_name)
);

CREATE TABLE eeg (
    recording_id INT NOT NULL,
    chan_name VARCHAR(80) NOT NULL,
    timepoint_sec DOUBLE NOT NULL,
    voltage DOUBLE, 
    FOREIGN KEY (recording_id)
        REFERENCES recordings(recording_id)
        ON DELETE CASCADE,
    PRIMARY KEY (recording_id, chan_name, timepoint_sec)
);

CREATE TABLE lfp (
    recording_id INT NOT NULL,
    chan_name VARCHAR(80) NOT NULL,
    timepoint_sec DOUBLE NOT NULL,
    voltage DOUBLE, 
    FOREIGN KEY (recording_id)
        REFERENCES recordings(recording_id)
        ON DELETE CASCADE,
    PRIMARY KEY (recording_id, chan_name, timepoint_sec)
);

CREATE TABLE temperature (
    recording_id INT NOT NULL,
    timepoint_sec DOUBLE NOT NULL,
    temperature_value DOUBLE,
    FOREIGN KEY (recordings_id)
        REFERENCES recordings(recording_id)
        ON DELETE CASCADE 
    PRIMARY KEY (recording_id, timepoint_sec)
);

CREATE TABLE neurons (
    neuron_id INT AUTO_INCREMENT PRIMARY KEY,
    cluster_id INT NOT NULL,
    recording_id INT NOT NULL,
    max_amp_channel INT,
    excluded INT,
    CONSTRAINT Unique_Neuron UNIQUE (cluster_id, recording_id),
    FOREIGN KEY (recording_id)
        REFERENCES recordings(recording_id)
        ON DELETE CASCADE
);

CREATE TABLE multi_units (
    mua_id INT AUTO_INCREMENT PRIMARY KEY,
    cluster_id INT,
    recording_id INT NOT NULL,
    FOREIGN KEY (recording_id)
        REFERENCES recordings(recording_id)
        ON DELETE CASCADE
);

CREATE TABLE neuron_clustering (
    neuron_id INT NOT NULL,
    grouping_method VARCHAR(250) NOT NULL,
    cluster VARCHAR(250),
    FOREIGN KEY (neuron_id)
        REFERENCES neurons(neuron_id)
        ON DELETE CASCADE
    PRIMARY KEY (neuron_id, grouping_method)
)

CREATE TABLE block_lengths (
    recording_id INT NOT NULL,
    block_name VARCHAR(150) NOT NULL,
    block_length BIGINT,
    PRIMARY KEY(recording_id, block_name)
);

CREATE TABLE good_spike_times (
    neuron_id INT NOT NULL,
    spike_times BIGINT NOT NULL,
    FOREIGN KEY (neuron_id)
        REFERENCES neurons(neuron_id)
        ON DELETE CASCADE,
    PRIMARY KEY(neuron_id, spike_times)
);

CREATE TABLE mua_spike_times (
    mua_id INT NOT NULL,
    spike_times BIGINT NOT NULL,
    FOREIGN KEY (mua_id)
        REFERENCES multi_units(mua_id)
        ON DELETE CASCADE,
    PRIMARY KEY(mua_id, spike_times)  
);

CREATE TABLE waveform_timepoints (
    neuron_id INT NOT NULL,
    sample INT NOT NULL,
    value DOUBLE,
    FOREIGN KEY(neuron_id)
        REFERENCES neurons(neuron_id)
        ON DELETE CASCADE,
    PRIMARY KEY (neuron_id, sample)
);

CREATE TABLE eshock_events (
    recording_id INT NOT NULL,
    event_sample INT NOT NULL,
    FOREIGN KEY (recording_id)
        REFERENCES recordings(recording_id)
        ON DELETE CASCADE,
    PRIMARY KEY(recording_id, event_sample)
);