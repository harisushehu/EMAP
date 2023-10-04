%% Load the set files and run PREP Pipeline
setPath = 'C:\Users\vh94rufa\Desktop\CleanDatasetB\filtered';
outPath = 'C:\Users\vh94rufa\Desktop\CleanDatasetB\icaed';

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

    %% ICA original
    % 20220729 took about 10 hours on matt's desktop comp. need to increase
    % max steps from 512
%     EEG = pop_runica(EEG, 'chanind',1:64,'extended',1,'stop',1e-8);  
    
    %% Find min(rank) among methods for PCA input. This is suggested by Makoto
    dataTempShort = EEG.data(1:64,1:3000);
    rankMATLAB = rank(dataTempShort);    
    [~, D] = eig(cov(dataTempShort', 1));
    rankHoffmann = sum (diag (D) > 1e-7);
    if rankMATLAB~=rankHoffmann
        fprintf('\n-----------------------------------------------------------------');
        fprintf('\nDifferent rank values calculated. MATLAB svd: %i; Hoffmann eigenvalue: %i. Using lower of these values as PCA input.', rankMATLAB, rankHoffmann);
        fprintf('\n-----------------------------------------------------------------\n');
    end
    
    %% ICA using Binica exe (https://github.com/cincibrainlab/Binica)
    EEG = pop_runica(EEG,'icatype','binica', 'chanind',1:64,'extended',1,'interupt','on','stop',1e-8,'maxsteps',1024,'pca',min(rankMATLAB,rankHoffmann));  
    
    
    
    %% save
    EEG = pop_saveset(EEG,'filename',pptSetFilename,'filepath',outPath);
        
end