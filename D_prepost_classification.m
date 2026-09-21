% ------------------------------------------------------------------------%
% SCRIPT D
% Five-fold linear SVM classification of baseline versus post-treatment.
% Uses every training-retained feature, not the descriptive top 40.
% Linear fitcsvm classifiers with within-participant label swaps.
% ------------------------------------------------------------------------%
clear; clc;
%% File paths
% Change as needed
hctsaFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/Revision_Analyses/D_Model_Corrections/PCA/Pilot/runtime_hctsa';
supportFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/support';


%% Inputs and settings
featureFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs/fivefold/features';
outputFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs/prepost_results';
numberOfPermutations = 1000;
seed = 20260909;
assert(~isfolder(outputFolder),'Choose a fresh output folder.');
folds = readtable(fullfile(featureFolder,'participant_folds.csv'),'TextType','string');
assert(height(folds)==42 && numel(unique(folds.Participant))==42);
assert(isequal(folds.BaselineRow,(1:42)') && isequal(folds.PostRow,(43:84)'));
assert(isequal(unique(folds.Fold),(1:5)'));
recordingFold = [folds.Fold;folds.Fold];
labels = [ones(42,1);2*ones(42,1)];

%% Start hctsa for its feature normalisation routines
previousFolder = pwd;
cd(hctsaFolder); clear startup; startup;
cd(previousFolder);
addpath(supportFolder);

%% Generate one common schedule of within-participant label swaps
rng(seed,'twister');
swapPairs = rand(42,numberOfPermutations)<0.5;
permutedLabels = repmat(labels,1,numberOfPermutations);
for permutation=1:numberOfPermutations
    people = find(swapPairs(:,permutation));
    permutedLabels(folds.BaselineRow(people),permutation) = 2;
    permutedLabels(folds.PostRow(people),permutation) = 1;
end
allLabels = [labels,permutedLabels];
predictions = nan(84,numberOfPermutations+1,3);
observedScores = nan(84,3);
retainedIDs = cell(5,3);

%% Prepare each fold, then fit the observed and shuffled models
for pc=1:3
    for fold=1:5
        loaded = load(fullfile(featureFolder,sprintf('features_fold%02d_PC%d.mat',fold,pc)));
        b = loaded.batch;
        isTraining = recordingFold~=fold;
        assert(all(b.completed) && b.fold==fold && b.pc==pc);
        assert(isequal(b.isTraining,isTraining));
        assert(isequal(b.recordingNames,[folds.BaselineLabel;folds.PostLabel]));
        [prepared,columns] = hctsa_prepare_fold(b.values,b.quality,isTraining);
        retainedIDs{fold,pc} = b.Operations.ID(columns);
        for permutation=0:numberOfPermutations
            y = allLabels(:,permutation+1);
            rng(seed+fold,'twister');
            model = fitcsvm(prepared(isTraining,:),y(isTraining),'KernelFunction','linear');
            [predicted,scores] = predict(model,prepared(~isTraining,:));
            predictions(~isTraining,permutation+1,pc) = predicted;
            if permutation==0
                observedScores(~isTraining,pc) = scores(:,model.ClassNames==2);
            end
        end
        fprintf('Finished fold %d PC%d.\n',fold,pc);
    end
end

%% Direct permutation p-values, then FDR across PCs
assert(all(isfinite(predictions),'all'));
correctCounts = squeeze(sum(predictions==allLabels,1));
observedAccuracy = correctCounts(1,:)'/84;
nullAccuracy = correctCounts(2:end,:)/84;
p = (1+sum(correctCounts(2:end,:)>=correctCounts(1,:),1))'/ ...
    (numberOfPermutations+1);
[sortedP,order] = sort(p);
adjusted = min(1,flipud(cummin(flipud(sortedP.*3./(1:3)'))));
pFDR = zeros(3,1);
pFDR(order) = adjusted;
results = table((1:3)',observedAccuracy,p,pFDR,'VariableNames', ...
    {'PC','BalancedAccuracy','P','PFDR'});

%% Save results 
mkdir(outputFolder);
writetable(results,fullfile(outputFolder,'summary.csv'));
save(fullfile(outputFolder,'classification_results.mat'),'results','predictions', ...
    'observedScores','nullAccuracy','allLabels','folds','retainedIDs','-v7.3');
disp(results);
