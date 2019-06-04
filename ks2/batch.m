%% set paths
addpath(genpath('/home/ruairi/repos/ephys_preprocessing/ks2'))

config_file = 'ks_config.m';
chan_map = 'chanMap_cam32.mat';

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

[names, ~] = deal(out{:});

for i = 1:length(names)
    names{i} = fullfile(data_path, names{i});
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
    todo = names{i};
    disp(todo)
 
    present = 0;
    for j = 1:length(names_done)
        to_check = names_done{j};
        if strcmp(todo, to_check)
            present = 1;
        end
    end
    if present == 1
        continue 
    end
    try
        kilosort2_fun(todo, config_file, chan_map);
        fout = log_out;
    catch ME
        disp('error')
        fout = didnt_work;
    end
    time = datestr(now);
    new_line = strjoin({todo, time, '\n'}, ',');
    log_out_fileid = fopen(fout, 'a');
    fprintf(log_out_fileid, new_line);
    fclose(log_out_fileid);
end

    
