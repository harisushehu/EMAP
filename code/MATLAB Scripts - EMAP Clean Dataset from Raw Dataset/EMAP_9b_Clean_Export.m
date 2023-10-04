%% 20210527 MJO Fixed failure to include *-session.csv file in clean dataset zips
%% 20220802 MJO: Not sure what I meant by the among. The session files (I believe) are in a separate zip, not included with each participants' trial data


clear all;

setPath = 'C:\Users\vh94rufa\Desktop\CleanDatasetB\iclabeled';
outPath = 'C:\Users\vh94rufa\Desktop\CleanDatasetB\csv files';
currPath = pwd; cd(setPath);
% [setFiles setPath] = uigetfile({'*.set','set files'},'Select the set files','MultiSelect', 'on');
% if ~iscellstr(setFiles), setFiles = {setFiles}; end
cd(currPath); 

[setFiles setPath] = uigetfile({'*.set','set files'},'Select the set files','MultiSelect', 'on');
if ~iscellstr(setFiles), setFiles = {setFiles}; end

for iPpt = 1:length(setFiles)
    pptSetFilename = setFiles{iPpt};
    whichParticipant = str2double(pptSetFilename(2:4));

    %% load .set file
    disp(['Loading data of: ' pptSetFilename]);
    clear EEG;
    EEG = pop_loadset('filename',pptSetFilename,'filepath',setPath);

    thisParticipantFiles = {};
    
    %% loop over each trial (from .set)
    for aTTN = unique(EEG.trialNumber(isfinite(EEG.trialNumber)))
        thisIndexMask = (EEG.trialNumber == aTTN);
        
        %% make a trial table
        clear tableToExport
        tableToExport = table(...
            repmat(whichParticipant,sum(thisIndexMask),1),...
            EEG.trialNumber(thisIndexMask)',...
            EEG.times(thisIndexMask)'/1000,...
            EEG.trialTimes(thisIndexMask)',...
            'VariableNames',...
            {'participant','trialNumber','sessionTime','trialTime'});

        %% add other measures
        tableToExport = [tableToExport,...
            array2table(EEG.data(1:64,thisIndexMask)','VariableNames',strcat(repmat({'EEG_'},1,64)',{EEG.chanlocs(1:64).labels}')),...
            array2table(EEG.data(65:66,thisIndexMask)','VariableNames',{'EOG_Vert','EOG_Horz'}),...
            table(...
            EEG.data_ecg(thisIndexMask)',...
            EEG.data_hr(thisIndexMask)',...
            EEG.data_gsr(thisIndexMask)',...
            EEG.data_irp(thisIndexMask)',...
            EEG.data_resp(thisIndexMask)',...
            EEG.data_response(thisIndexMask)',...
            'VariableNames',...
            {'ECG','heartrate','GSR','IRPleth','Respir','contArousal'})];
    
        %% export csv for each trial
        aFilename = ['P' num2str(whichParticipant,'%03.0f') '-T' num2str(aTTN,'%02.0f') '.csv'];
        disp(['Saving ' aFilename '...']);
        writetable(tableToExport,[outPath filesep aFilename]);
        thisParticipantFiles = [thisParticipantFiles aFilename];

    end
    
    %% compress the ~24 trial csvs into a single zip
    disp(['Compressing ' num2str(length(thisParticipantFiles)) ' files for P' num2str(whichParticipant,'%03.0f') '...']);
    zip([outPath filesep 'compressed' filesep 'P' num2str(whichParticipant,'%03.0f')], thisParticipantFiles, outPath);
end