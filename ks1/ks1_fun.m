function ks1_fun(data_file, data_root, config_file, chan_map)
    addpath(genpath('/home/ruairi/repos/KiloSort')) % path to kilosort folder
    addpath(genpath('/home/ruairi/repos/npy-matlab')) % path to npy-matlab scripts

    
    run(fullfile(config_file));

    tic
    if ops.GPU     
        gpuDevice(1); % initialize GPU (will erase any existing GPU arrays)
    end
    ops.fbinary             = data_file; % will be created for 'openEphys'		
    ops.fproc               = fullfile(data_root, 'temp_wh.dat'); % residual from RAM of preprocessed data		
    ops.root                = data_root; % 'openEphys' only: where raw files are	
    ops.chanMap             = chan_map;
    %
    [rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
    rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
    rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)
    
    % AutoMerge. rez2Phy will use for clusters the new 5th column of st3 if you run this)
    %     rez = merge_posthoc2(rez);
    
    % save matlab results file
    save(fullfile(ops.root,  'rez.mat'), 'rez', '-v7.3');
    
    % save python results file for Phy
    rezToPhy(rez, ops.root);
    
    % remove temporary file
    delete(ops.fproc); 
end