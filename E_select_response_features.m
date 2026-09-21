% ------------------------------------------------------------------------%
% SCRIPT E
% Select features within each LOPO training fold using hctsa.
% Normalised values rank/cluster the top 40; the response SVM uses the
% selected RAW baseline values. Select two per cluster, or one if singleton.
% This script does not use clinical response labels.
% ------------------------------------------------------------------------%
clear; clc;

%% File paths
hctsaFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/Revision_Analyses/D_Model_Corrections/PCA/Pilot/runtime_hctsa';
supportFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/support';

%% Inputs and selection settings
sourceFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs/lopo/features';
outputFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs/response_features';
numberOfFolds = 42;
numberTopFeatures = 40;
representativesPerCluster = 2;
distanceCutoff = 0.25;
seed = 0;
assert(~isfolder(outputFolder),'Choose a fresh output folder.');
folds = readtable(fullfile(sourceFolder,'participant_folds.csv'),'TextType','string');
assert(height(folds)==42 && numel(unique(folds.Participant))==42);
assert(isequal(folds.Fold,(1:42)'));
assert(isequal(folds.BaselineRow,(1:42)') && isequal(folds.PostRow,(43:84)'));

%% Start hctsa
previousFolder = pwd;
cd(hctsaFolder); clear startup; startup;
cd(previousFolder);
addpath(supportFolder);
rng(seed,'twister');
previousVisibility = get(groot,'defaultFigureVisible');
set(groot,'defaultFigureVisible','off');
mkdir(outputFolder);
writetable(folds,fullfile(outputFolder,'participant_folds.csv'));

%% Rank, cluster and select within each training fold

% Select representative features using hctsa within each training fold.
% hctsa's TS_TopFeatures ranks the features, and BF_pdist/BF_ClusterDown
% group the top 40 features by their similarity. We then select up to
% two highest-ranked features from each cluster.
% The selected raw baseline values are saved for the treatment-response SVM in Python.
% The hctsa features have already been calculated; this script selects
% which features to include in the response model.

for fold=1:numberOfFolds
    for pc=1:3
        loaded = load(fullfile(sourceFolder,sprintf('features_fold%02d_PC%d.mat',fold,pc)));
        b = loaded.batch;
        isTraining = b.isTraining;
        assert(all(b.completed) && b.fold==fold && b.pc==pc);
        assert(isequal(isTraining,[folds.Fold;folds.Fold]~=fold));
        assert(isequal(b.recordingNames,[folds.BaselineLabel;folds.PostLabel]));
    % Filter and normalise features using training recordings only.
    [normalised,eligibleColumns,replacements] = ...
        hctsa_prepare_fold(b.values,b.quality,isTraining);

    % 1. Rank the features using the training pre/post recordings.
    %hctsa needs the values, recording labels and feature names together.
    group = categorical([repmat("Baseline",42,1);repmat("Post",42,1)]);
    TimeSeries = table((1:84)',cellstr(b.recordingNames),group, ...
        'VariableNames',{'ID','Name','Group'});
    trainingData = struct('TS_DataMat',normalised(isTraining,:), ...
        'TimeSeries',TimeSeries(isTraining,:), ...
        'Operations',b.Operations(eligibleColumns,:));

    %This is hctsa's single-feature pre/post classification ranking.
    [~,rankingScores] = TS_TopFeatures(trainingData,'classification',struct(), ...
        'numTopFeatures',numberTopFeatures,'numNulls',0,'whatPlots',{});
    assert(all(isfinite(rankingScores)) && all(rankingScores>0), ...
        'Investigate a failed individual-feature classifier before continuing.');

    eligibleIDs = b.Operations.ID(eligibleColumns);
    % Highest score first; feature ID makes tied choices reproducible.
    [~,rankOrder] = sortrows([-rankingScores(:),eligibleIDs(:)],[1,2]);
    topLocalColumns = rankOrder(1:numberTopFeatures);
    topRawColumns = eligibleColumns(topLocalColumns);
    topIDs = b.Operations.ID(topRawColumns);
    topNames = string(b.Operations.Name(topRawColumns));
    topTraining = normalised(isTraining,topLocalColumns);

    % 2. Cluster the top 40 using the existing hctsa method.

    %Distance is 1 minus the magnitude of Spearman correlation.
    distances = BF_pdist(topTraining','abscorr');
    figuresBefore = findall(groot,'Type','figure');
    [~,clusterGroups] = BF_ClusterDown(distances, ...
        'clusterThreshold',distanceCutoff,'whatDistance','abscorr', ...
        'linkageMeth','average');
    close(setdiff(findall(groot,'Type','figure'),figuresBefore));

    % 3. Choose up to two features from each cluster.
    %Members are sorted by their position in the ranked top-40 list.
    clusterNumber = zeros(numberTopFeatures,1);
    chosen = false(numberTopFeatures,1);
    for clusterIndex = 1:numel(clusterGroups)
        members = sort(clusterGroups{clusterIndex}(:)); % rank order
        clusterNumber(members) = clusterIndex;
        numberToKeep = min(representativesPerCluster,numel(members));
        chosen(members(1:numberToKeep)) = true;
    end
    assert(all(clusterNumber>0));

    selectedColumns = topRawColumns(chosen);
    selectedIDs = topIDs(chosen);
    selectedNames = topNames(chosen);

    % 4. Prepare raw baseline values for the Python response SVM.
    % Normalisation above is for hctsa ranking/clustering only.
    % The response SVM uses these raw baseline values without extra scaling.

    baselineRows = folds.BaselineRow;
    rawBaseline = real(b.values(baselineRows,selectedColumns));
    validBaseline = isfinite(b.values(baselineRows,selectedColumns)) & ...
        imag(b.values(baselineRows,selectedColumns))==0 & ...
        b.quality(baselineRows,selectedColumns)==0;

    trainingMedians = median(real(b.values(isTraining,selectedColumns)),1);
    assert(all(validBaseline(folds.Fold~=fold,:),'all'));

    %Only invalid held-out baseline values can need replacement.
    [badRow,badColumn] = find(~validBaseline);
    for item = 1:numel(badRow)
        assert(folds.Fold(badRow(item))==fold);
        rawBaseline(badRow(item),badColumn(item)) = trainingMedians(badColumn(item));
    end
    assert(all(isfinite(rawBaseline),'all'));

        %% Save the selected baseline inputs and feature identities
        save(fullfile(outputFolder,sprintf('fold%02d_PC%d.mat',fold,pc)), ...
            'rawBaseline','isTraining','selectedIDs','selectedNames','eligibleIDs', ...
            'rankingScores','topIDs','chosen','clusterNumber','-v7');
    end
end
set(groot,'defaultFigureVisible',previousVisibility);
