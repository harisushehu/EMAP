sessions = readtable('V:\EMAP\Open Datasets\AllSessionData_20210205.csv');

for iP = 1:height(sessions)
    whichParticipant = sessions.participant(iP);
    try
        clear aSession;
        aSession = readtable(['V:\EMAP\Open Datasets\Raw Dataset\' num2str(whichParticipant,'P%03.0f') '-Session.csv']);
        
        for iF = {'Behaviour','EEG','Peripheral'}
            sessions.(['numTrialsWith' iF{1}])(iP) = sum(aSession.(['has' iF{1}]));
            
            sessions.(['missing' iF{1} 'Trials'])(iP) = {strjoin(cellstr(num2str(find(~aSession.(['has' iF{1}])))),',')};
        end
        
    
    
    
    
    
    catch
        % fill in filler
        disp(['Participant not found: ' num2str(whichParticipant,'P%03.0f')]);
    end
end

writetable(sessions,'V:\EMAP\Open Datasets\AllSessionData_20210205b.csv');
