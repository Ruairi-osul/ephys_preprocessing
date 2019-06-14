%% set paths
addpath(genpath('/home/ruairi/repos/ephys_preprocessing/ks1'))

config_file = 'ks1_config.m';
chan_map = 'chanMap.mat';

%didnt_work = '/media/ruairi/big_bck/CITWAY/log_files/problems.txt';
didnt_work = '/media/ruairi/big_bck/HAMILTON/log_files/problems.txt';

%data_path = /media/ruairi/big_bck/HAMILTON/probe_dat_dir;
data_path = '/media/ruairi/big_bck/HAMILTON/probe_dat_dir';
%% load log files

%log_out = '/media/ruairi/big_bck/CITWAY/log_files/kilosort.txt';
%log_in_path = '/media/ruairi/big_bck/CITWAY/log_files/pre_kilosort.txt';
log_out = '/media/ruairi/big_bck/HAMILTON/log_files/kilosort.txt';
log_in_path = '/media/ruairi/big_bck/HAMILTON/log_files/pre_kilosort.txt';
ks_path = log_out;

log_in_fileid = fopen(log_in_path, 'r');
out = textscan(log_in_fileid, '%s%s', 'delimiter', ',');
fclose(log_in_fileid);

[fnames, ~] = deal(out{:});

files = {};

for i = 1:length(fnames)
    names{i} = fullfile(data_path, fnames{i});
    files{i} = fullfile(names{i}, [fnames{i}, '.dat']);
end

log_in_fileid = fopen(ks_path, 'r');
out = textscan(log_in_fileid, '%s%s', 'delimiter', ',');
fclose(log_in_fileid);
[names_done, date] = deal(out{:});


%% single run

% config_file = fullfile(pwd, 'ks_config.m');
% chan_map = fullfile(pwd, 'chanMap_cam32.mat');
% todo = names{1};
% kilosort2_fun(todo, config_file, chan_map);
% 
% time = datestr(now);
% new_line = strjoin({todo, time, '\n'}, ',');
% log_out_fileid = fopen(log_out, 'a');
% fprintf(log_out_fileid, new_line);
% fclose(log_out_fileid);

%% batch
for i = 1:length(names)
    root = names{i};
    datfile = files{i};
    disp(root)
 
    present = 0;
    for j = 1:length(names_done)
        to_check = names_done{j};
        if strcmp(root, to_check)
            present = 1;
        end
    end
    if present == 1
        continue 
    end
    try
        ks1_fun(datfile, root, config_file, chan_map);
        fout = log_out;
    catch ME
        disp('error')
        fout = didnt_work;
    end
    time = datestr(now);
    new_line = strjoin({root, time, '\n'}, ',');
    log_out_fileid = fopen(fout, 'a');
    fprintf(log_out_fileid, new_line);
    fclose(log_out_fileid);
end

    
