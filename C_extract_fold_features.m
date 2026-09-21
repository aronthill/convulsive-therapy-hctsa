% ------------------------------------------------------------------------%
% SCRIPT C
% Apply each training PCA, then calculate the original hctsa features.
% Uses the unchanged TS_Compute through a small supporting routine.
% Run after B for fivefold, then for lopo. This is the expensive step.
% Starts fresh; it does not resume an interrupted extraction.
% Completed batches are saved in a new output folder.
% ------------------------------------------------------------------------%

clear; clc;
%% File paths

dataFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/data_from_g_drive/Combined_MST_ECT_Baseline_Post';
hctsaFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/Revision_Analyses/D_Model_Corrections/PCA/Pilot/runtime_hctsa';
supportFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/support';
seedHelperFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/Revision_Analyses/D_Model_Corrections/PCA/Pilot';
outputRoot = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs';

%% Choose inputs and a fresh output folder
validation = 'fivefold'; % 'fivefold' or 'lopo'
seed = 20260909;
samplesPerRecording = 4800;
pcaFolder = fullfile(outputRoot,validation,'pca');
outputFolder = fullfile(outputRoot,validation,'features');
assert(~isfolder(outputFolder),'Use a fresh output folder.');
p = load(fullfile(pcaFolder,'fold_assignment.mat'));
pairing = p.pairing;
recordingFold = p.recordingFold;
d = load(fullfile(dataFolder,'concatinated_data.mat'));
X = d.concatinated_data';
assert(isequal(size(X),[84*4800,60]) && all(isfinite(X),'all'));

%% Start the recorded hctsa runtime
previousFolder = pwd;
cd(hctsaFolder); clear startup; startup;
cd(previousFolder);
addpath(supportFolder);
addpath(seedHelperFolder); % Existing compiled random-seed helper.
assert(exist('hctsa_seed_c_rng','file')==3,'Build the native seed helper first.');
mkdir(outputFolder);
metadata = cell(3,1);
for pc=1:3
    metadata{pc} = load(fullfile(dataFolder,sprintf('HCTSA_PC%d_ec_zscore_by_individual_PCA',pc),'HCTSA.mat'), ...
        'TimeSeries','Operations','MasterOperations');
    names = string(metadata{pc}.TimeSeries.Name);
    assert(isequal(names(pairing.BaselineRow),pairing.BaselineLabel));
    assert(isequal(names(pairing.PostRow),pairing.PostLabel));
    assert(height(metadata{pc}.Operations)==7479); % Original TISEAN exclusion.
end

%% Project every recording and calculate its hctsa features
for fold=1:p.numberOfFolds
    fit = load(fullfile(pcaFolder,sprintf('pca_fold%02d.mat',fold)));
    isTraining = recordingFold~=fold;
    assert(isequal(fit.isTraining,isTraining));
    trainingMean = fit.trainingMean;
    assert(max(abs(mean(X(repelem(isTraining,4800),:),1)-trainingMean))<1e-10);
    for pc=1:3
        weights = fit.weights(:,pc);
        batch = struct('fold',fold,'pc',pc,'isTraining',isTraining, ...
            'Operations',metadata{pc}.Operations,'MasterOperations',metadata{pc}.MasterOperations, ...
            'recordingNames',string(metadata{pc}.TimeSeries.Name), ...
            'values',nan(84,7479),'quality',nan(84,7479),'completed',false(84,1));
        recordingOrder = [find(~isTraining);find(isTraining)]';
        for recording=recordingOrder
            sampleRows = (recording-1)*samplesPerRecording+(1:samplesPerRecording);
            inputSignal = (X(sampleRows,:)-trainingMean)*weights;
            r = extract_hctsa_recording(inputSignal,metadata{pc}.TimeSeries(recording,:), ...
                batch.Operations,batch.MasterOperations, ...
                fullfile(outputFolder,'current_recording.mat'),seed);
            batch.values(recording,:) = r.TS_DataMat;
            batch.quality(recording,:) = r.TS_Quality;
            batch.completed(recording) = true;
            valid = r.TS_Quality==0 & isfinite(r.TS_DataMat) & imag(r.TS_DataMat)==0;
            assert(mean(valid)>=0.8,'Investigate recording quality before continuing.');
        end
        save(fullfile(outputFolder,sprintf('features_fold%02d_PC%d.mat',fold,pc)), ...
            'batch','-v7.3');
        fprintf('Saved fold %d PC%d.\n',fold,pc);
    end
end
copyfile(fullfile(pcaFolder,'participant_folds.csv'),outputFolder);
