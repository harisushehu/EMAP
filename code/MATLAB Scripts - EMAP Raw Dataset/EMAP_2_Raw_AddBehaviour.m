
%% 2.1 Load Behaviour (in associated MAT file (with 'data_trial' structure))
clear Data_trial;
load([inPath filesep...
    ls([inPath filesep num2str(whichParticipant,'%3.0f') '_EMAP_*.mat'])],...
    'Data_trial');    

%% 2.3 Check that the data structures have 24 values. if not, some data needs to be removed so everything will line up properly.
% there's some complex input parsing logic here.
while any([length(EEG.fixationEventIndices) length(EEG.onsetEventIndices) length(EEG.offsetEventIndices) length(Data_trial)]-24)
%     error('skipped - fix either EEG events or Data_trial structure.');
    struct2table(EEG.event),
    Data_trial,

    disp('Paused - fix either EEG events or Data_trial structure.'); Beeper; 
    delEventRows = 'x'; delTrialRows = 'x';
    while ~isnumeric([delEventRows delTrialRows])
        try
            delEventRows = input('Enter EEG.event row indices (see above) to delete: (press enter to skip): ');
            delTrialRows = input('Enter Data_trial row indices (see above) to delete: (press enter to skip): ');
        catch
            delEventRows = 'x'; delTrialRows = 'x';
        end
    end
    EEG.event(delEventRows) = [];
    Data_trial(delTrialRows) = [];

    EEG.fixationEventIndices = find(strcmp({EEG.event.type},'Fixation'));
    EEG.onsetEventIndices = find(strcmp({EEG.event.type},'Onset'));
    EEG.offsetEventIndices = find(strcmp({EEG.event.type},'Offset'));

    trialRegistry = table(...
        (1:length(Data_trial))',...
        [Data_trial.thisTrialNumber]',...
        4*[Data_trial.movieDuration]',...
        'VariableNames',{'Index','TTN','DurationInBehaviour'}),
    try
        eegRegistry = table(...
            EEG.fixationEventIndices',...
            EEG.onsetEventIndices',...
            EEG.offsetEventIndices',...
            [EEG.event(EEG.onsetEventIndices).latency]' / EEG.srate,...
            ([EEG.event(EEG.offsetEventIndices).latency] / EEG.srate - [EEG.event(EEG.onsetEventIndices).latency] / EEG.srate)',...
            'VariableNames',{'FixationIndex','OnsetIndex','OffsetIndex','OnsetLatency','DurationInEEG'}),
    end
    
    disp('_____________________________');disp('Updated. New structures:');
    disp(['EEG has ' num2str(length(EEG.onsetEventIndices)) ' Onset Events;']);
    disp([num2str(whichParticipant,'%3.0f') '.mat has ' num2str(length(Data_trial)) ' trials of data;']);
    disp(['The duration mismatch (in sec): ' newline num2str(abs([eegRegistry.DurationInEEG]-[trialRegistry.DurationInBehaviour])') ]); 
    clear delEventRows delTrialRows
    
    if input('Looks good? Press 1 to continue.')
        break;
    end
end

%% 2.4 Instantiate new continual response field
EEG.data_response = nan(1,size(EEG.data,2));

%% 2.5 populate new event fields
for iT = 1:length(Data_trial)
    % iterate through the exp trials
    EEG.event(EEG.fixationEventIndices(iT)).trialNumber = Data_trial(iT).thisTrialNumber;
    EEG.event(EEG.onsetEventIndices(iT)).trialNumber = Data_trial(iT).thisTrialNumber;
    EEG.event(EEG.offsetEventIndices(iT)).trialNumber = Data_trial(iT).thisTrialNumber;

    %% 2.5.1 populate the data_response field with this trial's playback response
    % (remove the first sample of playback response; then interpolate to
    % EEG.srate)
    thisContinualResponse = interp1(...
        Data_trial(iT).playbackResponseTimestamps(2:end),...
        Data_trial(iT).playbackResponse(2:end),...
        Data_trial(iT).playbackResponseTimestamps(2) : 1/EEG.srate : Data_trial(iT).playbackResponseTimestamps(end),...
        'previous','extrap');
    EEG.data_response(EEG.event(EEG.onsetEventIndices(iT)).latency : EEG.event(EEG.onsetEventIndices(iT)).latency+length(thisContinualResponse)-1) = ...
        thisContinualResponse;
    clear thisContinualResponse;
end

   

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

