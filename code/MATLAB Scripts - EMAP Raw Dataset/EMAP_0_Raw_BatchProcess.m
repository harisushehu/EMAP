%% Wrapper to Batch process many ppt's data

%% Select multiple participant's data headers
% [inFiles inPath] = uigetfile({'EMAP_*.vhdr','First .vhdr file'},'Select the *.vhdr files','MultiSelect', 'on');
% if ~iscellstr(inFiles), inFiles = {inFiles}; end

% clear all; clc;
inPath = ['V:\EMAP\EMAP Data'];
outPath = ['V:\EMAP\Open Datasets\Raw Dataset'];

%% 
allLatencyDiffs = [];
failedParticipants = [];
allErrors = [];

%% loop
for whichParticipant = 118 %1:length(inFiles)
    clear EEG data_trial 
    
    if ~exist([inPath, filesep, 'EMAP_' num2str(whichParticipant,'%03.0f') '.vhdr'],'file') && ~exist([inPath, filesep, 'EMAP_' num2str(whichParticipant,'%04.0f') '.vhdr'],'file')
        disp(['Skipping #' num2str(whichParticipant,'%03.0f') ', no EEG data present...']);
        continue;
    else
        disp(['Loading #' num2str(whichParticipant,'%03.0f') '...']);
    end
    
    try
        EMAP_Raw_1_Preprocess;
        EMAP_Raw_2_AddBehaviour;
        EMAP_Raw_3_AddPeripheral;
        EMAP_Raw_4_AddTrialInfo;
        EMAP_Raw_5_Export;
    catch theError
        disp(['Failed on Participant #' num2str(whichParticipant) ', skipping.']);
        failedParticipants = [failedParticipants, whichParticipant];
        allErrors = [allErrors, theError];
        theError,
        Beeper(500,.5,.25);
    end

end