% Requires LabChart files to be exports as .mat (non-simple); a LabChart Macro can be created to do this in batch
% Takes in fully-processed EEG data in *readyToBin*.set files
% Finds corresponding LabChart-exported .mat file of psychophysio data
% co-registers the onset/offset events for each of these,
% adding in the psychophys data as new field in EEG

%% 3.1 load .mat file with labchart info
disp(['Loading pre-parsed LabChart data for ' num2str(whichParticipant) '...']);
clear lcEvents lcInputs
load(['V:\EMAP\Open Datasets\Labchart Data' filesep...
    num2str(whichParticipant,'%04.0f') '_parsed.mat']);    

%% 3.2 Instantiate new fields
EEG.data_ecg = nan(1,size(EEG.data,2));
EEG.data_hr = nan(1,size(EEG.data,2));
EEG.data_gsr = nan(1,size(EEG.data,2));
EEG.data_irp = nan(1,size(EEG.data,2));
EEG.data_resp = nan(1,size(EEG.data,2));

%% 3.3 add corresponding TTN
% more tricky logic here
lcEvents.CorrespEEGEventIndex = nan(height(lcEvents),1);
lcEvents.EEGEventLatency = nan(height(lcEvents),1);
for iEv = 1:length(EEG.event)
    if contains(EEG.event(iEv).type,{'Fixation','Onset','Offset'})
        lcIndex = find(contains(lcEvents.Comment,EEG.event(iEv).type) & lcEvents.AssumedTTN==EEG.event(iEv).trialNumber);
        lcEvents.CorrespEEGEventIndex(lcIndex) = iEv;
        lcEvents.EEGEventLatency(lcIndex) = EEG.event(iEv).latency;
    end
end
lcEvents.TimeDiffFromEEG = [NaN;diff(lcEvents.EEGEventLatency)]/EEG.srate;
lcEvents.LatencyError = lcEvents.TimeDiff - lcEvents.TimeDiffFromEEG;

% if sum(abs(lcEvents.LatencyError)>=.004)~=1 || ~contains(lcEvents.Comment(abs(lcEvents.LatencyError)>=.004),'Fixation') || lcEvents.AssumedTTN(abs(lcEvents.LatencyError)>=.004)~=13 
    weirdEvents = lcEvents(abs(lcEvents.LatencyError)>=.004,:),
    allLatencyDiffs = [allLatencyDiffs;...
        [table(repmat(whichParticipant,height(weirdEvents),1),'VariableNames',{'Participant'}), weirdEvents]];
%     Beeper;
%     input('Check the LatencyErrors above; press a key to continue or Ctrl+C to break and correct.');
% end

%% 3.3.1 Ensure there are 72 Labchart events ("comments"); if not matching with other data, remove some events ("comments")
while height(lcEvents)~=length([EEG.onsetEventIndices, EEG.offsetEventIndices, EEG.fixationEventIndices])
    lcEvents,
    disp('Different numbers of events in EEG and LC data. Remove some event indices.'); Beeper; 
    
    % manually assign the LC triggers, by looking carefully at the comments in labchart and collating those with the eeg data...
    delEventRows = 'x';
    while ~isnumeric(delEventRows)
        try
            delEventRows = input('Enter COMMENT NUMBERS (see above) to delete (i.e. "[1,6]"; press enter to skip): ');
        catch
            delEventRows = 'x';
        end
    end
    lcEvents(ismember(lcEvents.CommentNumber,delEventRows),:) = [],
end

%% 3.4 Adding psy-phy data to EEG struct. 
% This is tricky, because the EEG data has gaps (due to "boundary" events).
% The psychophysiology has to be added to that timeline, while being aware 
% of those gaps. BUT, the psyphy also has gaps, but it's devided into
% blocks. The psyphy data may overrun the EEG data or come short. So, I am
% handling it in a very brute force and slow way.
% when reaching a boundary in EEG data, will stop populating. 

%% find the "seed" rows, which connect the eeg and psyphy. will populate the data starting at these points.
seedRows = [];
boundaryEventFrames = [];
% loop over blocks of lc data. 
% for each block, find the first onset events common to both datasets
for whichBlock = unique(lcEvents.Block)'
    seedRows = [seedRows,...
        find([lcEvents.Block]==whichBlock & contains(lcEvents.Comment,'Onset') & [lcEvents.CorrespEEGEventIndex]>0,1,'first')];
end
% for each boundary event, find onset events (EEG Index) common to both
for whichBoundaryEventIndex = find(strcmp({EEG.event.type},'boundary'))
    seedRows = [seedRows,...
        find([lcEvents.CorrespEEGEventIndex]>=whichBoundaryEventIndex & contains(lcEvents.Comment,'Onset'),1,'first')];
    
    % by the way, record the latency of boundary events so we can check for
    % them later
    boundaryEventFrames = [boundaryEventFrames EEG.event(whichBoundaryEventIndex).latency];
end

%% 3.4 Loop over each seed row in lcEvents
for aSeedRow = unique(seedRows)
    
    %% indices of data to populate
    sourceStartEndIndices = [];
    destStartEndIndices = [];
    
    %% 3.4.2 Loop over forward and backward directions
    for populateDirection = [-1,1]
        disp(['Populating from lcEvent #' num2str(aSeedRow) ' in direction ' num2str(populateDirection) '.']);
        
        %% 3.4.2.1 Find the corresponding EEG frame to begin populating
        currentFrameSource = lcEvents.DataIndexOffset(aSeedRow)+3; % assumes downsampling by 4x; also offset by 3 points
        currentFrameDest = lcEvents.EEGEventLatency(aSeedRow);
        
        %% 3.4.2.2 While loop: populate starting from source event
        while 1
            % check that haven't reached an EEG boundary event:
            if ismember(currentFrameDest,boundaryEventFrames) 
                disp(['~ Stopped at EEG time ' num2str(currentFrameDest/(60*250),'%0.1f') 'min; reached EEG boundary event.']);
                break; % if so, stop populating in this direction.
            % check that we haven't reached the limit of EEG data (theres a
            % frame available to fill in EEG):
            elseif currentFrameDest>=length(EEG.data) || currentFrameDest<=1 
                disp(['~ Stopped at EEG time ' num2str(currentFrameDest/(60*250),'%0.1f') 'min; reached limit of EEG data frames.']);
                break; % if so, stop populating in this direction.
            % check that there is a frame available in Psyphy data:
            elseif currentFrameSource<=4 || currentFrameSource>=(size(lcInputs{lcEvents.Block(aSeedRow)}.Data,2)-3) 
                disp(['~ Stopped at EEG time ' num2str(currentFrameDest/(60*250),'%0.1f') 'min; reached limit of psychophysio data frames.']);
                break; % if not, stop populating in this direction.
            end
            
            %% else, move the data pointer
            currentFrameSource = currentFrameSource + populateDirection*4; % assumes x4 downsample
            currentFrameDest = currentFrameDest + populateDirection*1;
        end
        
        sourceStartEndIndices = [sourceStartEndIndices, currentFrameSource];
        destStartEndIndices = [destStartEndIndices, currentFrameDest];
        
        if diff(sourceStartEndIndices)/4 ~= diff(destStartEndIndices)
            error('different size source and destination data ranges!');
        end
        
    end

    %% Extract psyphy data for each trial and copy to EEG struct
    % NB - we effectively downsample by populating with every 4th data
    % point. (and iterate data source pointer by 4)
    sensorNames = {'ecg','hr','gsr','irp','resp'};
    for sensorIndex = 1:length(sensorNames)
        tempSample = lcInputs{lcEvents.Block(aSeedRow)}.Data(sensorIndex,sourceStartEndIndices(1):4:sourceStartEndIndices(2));
        EEG.(['data_' sensorNames{sensorIndex}])(destStartEndIndices(1):destStartEndIndices(2)) = tempSample;
    end
    clear tempSample;
end
