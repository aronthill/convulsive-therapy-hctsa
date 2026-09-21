% ------------------------------------------------------------------------%
% SCRIPT B
% Fit PCA using training participants only.
% Use MATLAB's pca function and retain its component signs.
% Run with validation='fivefold' for pre/post, then 'lopo' for response.
% Input EEG has ALREADY been resampled and standardised within each epoch.
% ------------------------------------------------------------------------%

close all; clear; clc;
%% File paths

% Set filepaths
dataFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/data_from_g_drive/Combined_MST_ECT_Baseline_Post';
pairingFile = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/Revision_Analyses/A_Paired_Statistics/results/preparation_20260908_170400_468/participant_pairing.csv';
outputRoot = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs';


%% Choose the validation scheme
validation = 'fivefold'; % 'fivefold' or 'lopo'
seed = 20260909;
samplesPerRecording = 4800;
outputFolder = fullfile(outputRoot,validation,'pca');
assert(ismember(validation,{'fivefold','lopo'}));
assert(~isfolder(outputFolder),'Choose a new output folder.');
assert(contains(which('pca'),fullfile(matlabroot,'toolbox','stats')));

%% Load EEG and the checked recording order
pairing = readtable(pairingFile,'TextType','string');
assert(height(pairing)==42 && numel(unique(pairing.Participant))==42);
assert(isequal(pairing.BaselineRow,(1:42)') && isequal(pairing.PostRow,(43:84)'));
d = load(fullfile(dataFolder,'concatinated_data.mat'));
X = d.concatinated_data'; % Timepoints in rows; 60 electrodes in columns.
assert(isequal(size(X),[84*samplesPerRecording,60]) && all(isfinite(X),'all'));
for pc=1:3
    m = load(fullfile(dataFolder,sprintf('HCTSA_PC%d_ec_zscore_by_individual_PCA',pc),'HCTSA.mat'),'TimeSeries');
    names = string(m.TimeSeries.Name);
    assert(isequal(names(pairing.BaselineRow),pairing.BaselineLabel));
    assert(isequal(names(pairing.PostRow),pairing.PostLabel));
end

%% Keep both recordings from each participant in the same fold
if strcmp(validation,'fivefold')
    rng(seed,'twister');
    participantFold = zeros(42,1);
    treatmentNames = ["ECT","MST"];
    for treatment=1:2
        rows = find(pairing.Treatment==treatmentNames(treatment));
        assert(numel(rows)==21);
        rows = rows(randperm(21));
        participantFold(rows) = mod((0:20)'+treatment-1,5)+1;
    end
else
    participantFold = (1:42)'; % Each person is held out once.
end
recordingFold = zeros(84,1);
recordingFold(pairing.BaselineRow) = participantFold;
recordingFold(pairing.PostRow) = participantFold;
numberOfFolds = max(participantFold);
mkdir(outputFolder);
folds = pairing;
folds.Fold = participantFold;
writetable(folds,fullfile(outputFolder,'participant_folds.csv'));

%% Fit each PCA on the training recordings
for fold=1:numberOfFolds
    isTraining = recordingFold~=fold;
    trainingSamples = repelem(isTraining,samplesPerRecording);
    [weights,~,~,~,explained,trainingMean] = ...
        pca(X(trainingSamples,:),'NumComponents',10);
    save(fullfile(outputFolder,sprintf('pca_fold%02d.mat',fold)), ...
        'weights','trainingMean','explained','isTraining','fold');
    fprintf('PCA fold %d/%d complete.\n',fold,numberOfFolds);
end
save(fullfile(outputFolder,'fold_assignment.mat'),'pairing','recordingFold', ...
    'participantFold','numberOfFolds','seed','validation');
