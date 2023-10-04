%% Load the set files and run ICLabel
setPath = 'C:\Users\vh94rufa\Desktop\CleanDatasetB\icaed';
outPath = 'C:\Users\vh94rufa\Desktop\CleanDatasetB\iclabeled';

currPath = pwd; cd(setPath);
[setFiles setPath] = uigetfile({'*.set','set files'},'Select the set files','MultiSelect', 'on');
if ~iscellstr(setFiles), setFiles = {setFiles}; end
cd(currPath); 

failedLoads = {};
compsRemoved = [];

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
    
    %% ICLabel
  	EEG = iclabel(EEG);  
    
    high_artifact = any(EEG.etc.ic_classification.ICLabel.classifications(:,2:6)>.8 , 2);
    low_brain = EEG.etc.ic_classification.ICLabel.classifications(:,1)<.1;
    EEG.reject.gcompreject = and(high_artifact,low_brain);
    fprintf('To reject %i comps: ', sum(EEG.reject.gcompreject)); fprintf('%i, ' , find(EEG.reject.gcompreject)); disp(' ');

%     pop_viewprops(EEG,0,find(EEG.reject.gcompreject)','freqrange', [2 50]);
%     EEG = pop_selectcomps(EEG, 1:35);
%     Beeper; pause(0.01); figh = gcf; %frame_h = get(handle(gcf),'JavaFrame'); %set(frame_h,'Maximized',1);
    % manually choose and reject components in GUI...
    % wait until user has closed the component selection window, then remove components. 
%     while ishandle(figh)
%         drawnow;
%     end

    compsRemoved(iSet,:) = [pptNum, sum(EEG.reject.gcompreject)];
    
    EEG = pop_subcomp( EEG, [], 0);

    
    %% save
    EEG = pop_saveset(EEG,'filename',pptSetFilename,'filepath',outPath);
        
end

save('rejectionStats.mat','compsRemoved');