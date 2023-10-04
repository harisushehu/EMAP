%% Preprocessor for EMAP (Hedwig, Tim, Harisu)
% Written by Matt
% Requires EEGLAB
% Requires PREP Pipeline
% Intakes .vhdr files and does the first part of processing of these.
% Outputs a partially processed .set file, ready for collation with
% behavioural data.
% 20210130 MJO Final version 

%% 1.1: Load BrainVision EEG data into 'EEG'
if ~exist('whichParticipant','var')
    whichParticipant = input('Process which participant? (enter number): ');
end

if whichParticipant<10
    EEG = pop_loadbv(inPath, ['EMAP_' num2str(whichParticipant,'%03.0f') '.vhdr']);
else
    EEG = pop_loadbv(inPath, ['EMAP_' num2str(whichParticipant,'%04.0f') '.vhdr']);
end
    
%% 1.2 Add channel location information and re-add FCz channel. 
EEG = pop_chanedit(EEG, 'setref',{'1:63' 'FCz'});
EEG = pop_chanedit(EEG,...
    'insert',64,'changefield',{64 'labels' 'FCz'},...
    'lookup','standard-10-5-cap385.elp');

%% 1.3 Average reference, then reorder channels for sanity. EOG are now 65 & 66
EEG = pop_reref( EEG, [], 'exclude', [64 65],...
    'refloc',struct('labels',{'FCz'},'type',{''},'theta',{0},'radius',{0.12662},'X',{32.9279},'Y',{0},'Z',{78.363},'sph_theta',{0},'sph_phi',{67.208},'sph_radius',{85},'urchan',{64},'ref',{''},'datachan',{0}));
EEG.chanlocs = EEG.chanlocs([1:63,66,64,65]);
EEG.data = EEG.data([1:63,66,64,65],:);


%% 1.4 Rename events corresponding to the onsets/offsets in each video
for iEvent = 1:length(EEG.event)
    t = {EEG.event(iEvent).type}; % sometimes multiple types appear in a single event
    switch t{1}
        case 'S  2'
            EEG.event(iEvent).type = 'Fixation';
        case 'S  4'
            EEG.event(iEvent).type = 'Onset';
        case 'S  8'
            EEG.event(iEvent).type = 'Offset';
        case 'boundary'
            EEG.event(iEvent).type = 'boundary';
        otherwise
            EEG.event(iEvent).type = 'Remove';                
    end
    clear t;
end
EEG.event(strcmp({EEG.event.type},'Remove')) = []; % remove extraneous events
summary(categorical({EEG.event.type})'),


EEG.fixationEventIndices = find(strcmp({EEG.event.type},'Fixation'));
EEG.onsetEventIndices = find(strcmp({EEG.event.type},'Onset'));
EEG.offsetEventIndices = find(strcmp({EEG.event.type},'Offset'));

%% 1.5 Remove 10 seconds before first/after last trial
firstPoint = EEG.event(EEG.fixationEventIndices(1)).latency - 10*EEG.srate;
lastPoint = EEG.event(EEG.offsetEventIndices(end)).latency + 10*EEG.srate;
EEG = pop_select( EEG,'point', [ firstPoint lastPoint ]);
clear firstPoint lastPoint

%% 1.6 Downsample to 250
EEG = pop_resample( EEG, 250);

%% 1.6.1
% experimental: Floor all latencies to get rid of fractional frames
% delete 0 latency events (just a boundary)
for iEvent = 1:length(EEG.event)
    EEG.event(iEvent).latency = floor(EEG.event(iEvent).latency);
end

%% delete 0 latency events (just a boundary) - then have to find out the indices again!
EEG.event(strcmp({EEG.event.type},'boundary') & [EEG.event.latency]<=0) = []; 
EEG.fixationEventIndices = find(strcmp({EEG.event.type},'Fixation'));
EEG.onsetEventIndices = find(strcmp({EEG.event.type},'Onset'));
EEG.offsetEventIndices = find(strcmp({EEG.event.type},'Offset'));

%% save
% EEG = pop_saveset(EEG,'filename',num2str(whichParticipant,'%03.0f'),'filepath',outPath);
