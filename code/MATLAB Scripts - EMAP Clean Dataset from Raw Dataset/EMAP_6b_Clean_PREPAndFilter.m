%% Load the set files and run PREP Pipeline

setPath = 'D:\2021 Hedwig EMAP\2021 Analysis\Raw Dataset\EEGLAB set files';
outPath = 'D:\2021 Hedwig EMAP\2022 Analysis Redo Clean Dataset\Clean Dataset B';
currPath = pwd; cd(setPath);
[setFiles setPath] = uigetfile({'*.set','set files'},'Select the set files','MultiSelect', 'on');
if ~iscellstr(setFiles), setFiles = {setFiles}; end
cd(currPath); 

failedLoads = {};

for iSet = 1:length(setFiles)
    pptSetFilename = setFiles{iSet};
    pptNum = str2double(pptSetFilename(2:4));
    
    try
        %% load .set file
        disp(['Loading data of: ' pptSetFilename]);
        clear EEG;
        EEG = pop_loadset('filename',pptSetFilename,'filepath',setPath);
        
    catch
        warning([pptSetFilename ' could not be loaded. Continuing.']);
        failedLoads = [failedLoads, pptSetFilename];
        continue;
    end
        
    %% PREP_pipeline & bandpass
    % should be ok to run prep with boundary events (block start/stops), if the several seconds on
    % either side of the boundaries are not important: https://sccn.ucsd.edu/pipermail/eeglablist/2015/010268.html
    % 20220729: The code is the same as used for Matt's RoleReversal EEG
    % and checked by Norman Forschack.
    [EEG, ~] = prepPipeline(EEG,...
        struct('lineFrequencies',[50 100 150 200],...
        'ignoreBoundaryEvents',true,...
        'referenceChannels',1:64,...
        'rereferencedChannels',1:64,...
        'evaluationChannels',1:64));
    EEG.etc.noiseDetection.errors.status,
    
    %% Filtering (redone 20220729)
    % highpass band changed to 0.5 and using proper filtering method.
    % Filter code borrowed from Norman Forschack, 20220607.
    % DEP: EEG = pop_eegfiltnew(EEG, 48,52,826,1,[],0); % notch 48-52 Hz
    % DEP: EEG = pop_eegfiltnew(EEG, 2,100,826,0,[],0); % bandpass 2-100 Hz
    
    % Lowpass filter
    filters.lp_cutoff = 100; % hz
    filters.lp_maxPBD = 0.0001; % max passband deviation
    filters.lp_TBW = 0.25*filters.lp_cutoff; % thresold bandwidth; for cutoffs > 8 Hz should be 25% of cutoff (remember cutoff is centered (-6dB) within transition bandwidth)
    filters.lp_order = pop_firwsord('kaiser', EEG.srate, filters.lp_TBW, filters.lp_maxPBD);
    fprintf('Lowpass filtering activity above %0.1fhz...\n', filters.lp_cutoff);    
    EEG = pop_firws(EEG,...
        'fcutoff',filters.lp_cutoff,...
        'ftype','lowpass',...
        'wtype','kaiser','warg',pop_kaiserbeta(filters.lp_maxPBD),...
        'forder',filters.lp_order,'minphase',0);
    
    % Highpass filter
    filters.hp_cutoff = .5; % hz; widmann suggests <= 0.1Hz
    filters.hp_maxPBD = 0.001; % max passband deviation
    filters.hp_TBW = 2*filters.hp_cutoff; % thresold bandwidth; 2*cutoff for cutoffs<=1
    filters.hp_order = pop_firwsord('kaiser', EEG.srate, filters.hp_TBW, filters.hp_maxPBD);
    fprintf('Highpass filtering activity below %0.1fhz...\n', filters.hp_cutoff);    
    EEG = pop_firws(EEG,...
        'fcutoff',filters.hp_cutoff,...
        'ftype','highpass',...
        'wtype','kaiser','warg',pop_kaiserbeta(filters.hp_maxPBD),...
        'forder',filters.hp_order,'minphase',0);

    %% ICA
    % 20220729 took about 10 hours on matt's desktop comp. need to increase
    % max steps from 512
%     EEG = pop_runica(EEG, 'chanind',1:64,'extended',1,'stop',1e-8);  
    
    %% save
    EEG = pop_saveset(EEG,'filename',[pptSetFilename '-filtered'],'filepath',[outPath filesep 'filtered']);
        
end