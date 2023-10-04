% Requires LabChart files to be exports as .mat (non-simple); a LabChart Macro can be created to do this in batch
% Takes in fully-processed EEG data in *readyToBin*.set files
% Finds corresponding LabChart-exported .mat file of psychophysio data
% co-registers the onset/offset events for each of these,
% adding in the psychophys data as new field in EEG

%% 3.1 load .mat file with labchart info
disp(['Loading LabChart data for ' num2str(whichParticipant) '...']);
lc = load([inPath filesep '..' filesep 'EEG Processed' filesep 'LabChart Exported' filesep...
    num2str(whichParticipant,'%04.0f') '.mat']);    
Tool_LabchartParser; 
clear lc;

%% 2.4 Instantiate new fields
EEG.data_ecg = nan(1,size(EEG.data,2));
EEG.data_hr = nan(1,size(EEG.data,2));
EEG.data_gsr = nan(1,size(EEG.data,2));
EEG.data_irp = nan(1,size(EEG.data,2));
EEG.data_resp = nan(1,size(EEG.data,2));

%% 3.2 Ensure there are 72 Labchart events ("comments"); if not matching with other data, remove some events ("comments")
while height(lcEvents)~=length([EEG.onsetEventIndices, EEG.offsetEventIndices, EEG.fixationEventIndices])
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
        
%% 3.3 step through each trial (from .set)
for iT = 1:length(EEG.onsetEventIndices)
    disp([num2str(iT) ': parsing trial #' num2str(EEG.event(EEG.onsetEventIndices(iT)).trialNumber) ' data of participant #' num2str(whichParticipant)]);
    clear dt; 
    dt = EEG.event(EEG.onsetEventIndices(iT));
%     dt.numFrames = ceil( EEG.event(EEG.offsetEventIndices(iT)).latency - EEG.event(EEG.fixationEventIndices(iT)).latency-1 );
%     dt.times = allTimes(1:dt.numFrames);
    
    %% find fixation and end data points for psychophysio data extraction
    % Method 1: Use trial number as labeled in LC comment text
    fixationBlock = lcEvents{strcmp(lcEvents.Comment,['Fixation Trial ' num2str(EEG.event(EEG.onsetEventIndices(iT)).trialNumber)]),'Block'};
    fixationOffset = lcEvents{strcmp(lcEvents.Comment,['Fixation Trial ' num2str(EEG.event(EEG.onsetEventIndices(iT)).trialNumber)]),'DataIndexOffset'};
    offsetOffset = lcEvents{strcmp(lcEvents.Comment,['Offset Trial ' num2str(EEG.event(EEG.onsetEventIndices(iT)).trialNumber)]),'DataIndexOffset'};
    
    % Method 2: Use LC comments in order, assuming no missing events
    % To use if there's a line-up issue
%     fixationRows = find(contains(lcEvents.Comment,'Fixation Trial'));
%     offsetRows = find(contains(lcEvents.Comment,'Offset Trial'));
%     fixationBlock = lcEvents{fixationRows(iT),'Block'};
%     fixationOffset = lcEvents{fixationRows(iT),'DataIndexOffset'};
%     offsetOffset = lcEvents{offsetRows(iT),'DataIndexOffset'};

    %% what about all the peripheral data BETWEEN trials?!
        
    %% check that labchart data is appropriate length
    if ((offsetOffset-fixationOffset)/1000 > 90) || ((offsetOffset-fixationOffset)/1000 < 30)
%         (EEG.event(EEG.offsetEventIndices(iT)).latency - EEG.event(EEG.onsetEventIndices(iT)).latency)>=(90*1000/4)
        disp(['--- Skipped, duration of LC data was ' num2str((offsetOffset-fixationOffset)/1000) ' seconds...?']); 
        continue;
    end
        
    %% Extract psyphy data for each trial, downsample to 250 hz and copy to EEG struct
    sensorNames = {'ecg','hr','gsr','irp','resp'};
    for sensorIndex = 1:length(sensorNames)
        tempWhat = downsample(...
            lcInputs{fixationBlock}.Data(sensorIndex,fixationOffset:offsetOffset),...
            lcInputs{fixationBlock}.SampleRate(sensorIndex) / EEG.srate, (lcInputs{fixationBlock}.SampleRate(sensorIndex) / EEG.srate)-1);
        tempWhere = ceil(EEG.event(EEG.fixationEventIndices(iT)).latency + (1:length(tempWhat)) - 1);
        EEG.(['data_' sensorNames{sensorIndex}])(tempWhere) = tempWhat;
    end
        
end


