function done = kilosort2_fun(data_path, config_path, chanmap_path)
    kilosort2_path = '/home/ruairi/repos/Kilosort2';
    npy_path = '/home/ruairi/repos/npy-matlab';

    addpath(npy_path);
    addpath(genpath(kilosort2_path));
    run(config_path);
    rootH = data_path;
    rootZ = rootH;
    ops.fproc = fullfile(rootH, 'temp_wh.dat');
    ops.chanMap = chanmap_path;
    ops.trange = [0 Inf]; % time range to sort
    ops.NchanTOT    = 32; % total number of channels in your recording

    fs = [dir(fullfile(rootZ, '*.bin')) dir(fullfile(rootZ, '*.dat'))];
    ops.fbinary = fullfile(rootZ, fs(1).name);
    rez = preprocessDataSub(ops);
    rez = clusterSingleBatches(rez);
    save(fullfile(rootZ, 'rez.mat'), 'rez', '-v7.3');
    rez = learnAndSolve8b(rez);
    rez = find_merges(rez, 1);
    rez = splitAllClusters(rez, 1);
    rez = splitAllClusters(rez, 0);
    rez = set_cutoff(rez);
    fprintf('found %d good units \n', sum(rez.good>0))
    fprintf('Saving results to Phy  \n')
    rezToPhy(rez, rootZ);
    rez.cProj = [];
    rez.cProjPC = [];
    fprintf('Saving final results in rez2  \n')
    fname = fullfile(rootZ, 'rez2.mat');
    save(fname, 'rez', '-v7.3');
    done = 1;
end