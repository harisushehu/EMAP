EEG.trialTimes = nan(1,size(EEG.data,2));
EEG.trialNumber = nan(1,size(EEG.data,2));

%% add trial times
for i = 1:length(EEG.onsetEventIndices)
    % again, populating forward and backward from onset so that any minor
    % timing issues with events are worst at fixation and end of the trial,
    % not at the video onset.
    
    %populate prestim times
    framesToPopulate = EEG.event(EEG.fixationEventIndices(i)).latency : EEG.event(EEG.onsetEventIndices(i)).latency ;
    timesToPopulate = fliplr( -4 * (0:1:length(framesToPopulate)-1) / 1000 );
    EEG.trialTimes(framesToPopulate) = timesToPopulate;
    EEG.trialNumber(framesToPopulate) = repmat(EEG.event(EEG.onsetEventIndices(i)).trialNumber,1,length(framesToPopulate));    
     
    %populate poststim times
    framesToPopulate = EEG.event(EEG.onsetEventIndices(i)).latency : EEG.event(EEG.offsetEventIndices(i)).latency ;
    timesToPopulate = 4 * (0:1:(length(framesToPopulate)-1)) / 1000;
    EEG.trialTimes(framesToPopulate) = timesToPopulate;
    EEG.trialNumber(framesToPopulate) = repmat(EEG.event(EEG.onsetEventIndices(i)).trialNumber,1,length(framesToPopulate));    

end
   
%% trialInfo struct array
% this will use the "data_trial" which has been subsetted to match EEG data
% present. As a result, there are a few more trials of behavioural data across the whole
% set that won't be included here. If these are to be included, the
% original Data_trials will need to be reloaded here for parsing.
clear trialInfo
trialInfo = table;
trialInfo.participant = [Data_trial.Participant]';
trialInfo.trialNumber = [Data_trial.thisTrialNumber]';
trialInfo.videoCode = [Data_trial.videoCode]';
trialInfo.movieArousal = {Data_trial.movieArousal}';
trialInfo.movieValence = {Data_trial.movieValence}';
trialInfo.respArousal = [Data_trial.respArousal]';
trialInfo.respValence = [Data_trial.respValence]';
trialInfo.respLiking = [Data_trial.respLiking]';
trialInfo.respApproach = [Data_trial.respApproach]';
trialInfo.respAnger = [Data_trial.respAnger]';
trialInfo.respHappy = [Data_trial.respHappy]';
trialInfo.respSadness = [Data_trial.respSadness]';
trialInfo.respDisgust = [Data_trial.respDisgust]';
trialInfo.respFear = [Data_trial.respFear]';
trialInfo.fixationTime = EEG.times([EEG.event(EEG.fixationEventIndices).latency])'/1000;
trialInfo.onsetTime = EEG.times([EEG.event(EEG.onsetEventIndices).latency])'/1000;
trialInfo.offsetTime = EEG.times([EEG.event(EEG.offsetEventIndices).latency])'/1000;
trialInfo.hasBehaviour = 1*isfinite(EEG.data_response(1,[EEG.event(EEG.onsetEventIndices).latency]))';
trialInfo.hasEEG = 1*isfinite(EEG.data(1,[EEG.event(EEG.onsetEventIndices).latency]))';
trialInfo.hasPeripheral = 1*isfinite(EEG.data_ecg(1,[EEG.event(EEG.onsetEventIndices).latency]))';