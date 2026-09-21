% ------------------------------------------------------------------------%
% SCRIPT A
% Paired baseline/post feature comparisons.
% hctsa's original feature test did not account for participant pairing.
% Retain the original normalised features and use MATLAB signrank FDR.
% Run separately for PCs 1, 2 and 3. No feature ranking is changed here.
% ------------------------------------------------------------------------%

close all; clear; clc;
%% File paths
% Change these paths if running on another computer.
dataFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/data_from_g_drive/Combined_MST_ECT_Baseline_Post';

pairingFile = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/Revision_Analyses/A_Paired_Statistics/results/preparation_20260908_170400_468/participant_pairing.csv';

hctsaFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/Revision_Analyses/D_Model_Corrections/PCA/Pilot/runtime_hctsa';

outputRoot = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/outputs';


%% Choose PC to examine
pc = 1;
fdrThreshold = 0.05;
addpath(fullfile(hctsaFolder,'PeripheryFunctions'));
inputFile = fullfile(dataFolder,sprintf('HCTSA_PC%d_ec_zscore_by_individual_PCA',pc),'HCTSA_N.mat');
[TS_DataMat,TimeSeries,Operations] = TS_LoadData(inputFile);
pairing = readtable(pairingFile,'TextType','string');

%% Check recording identities before pairing
assert(height(pairing)==42 && numel(unique(pairing.Participant))==42);
assert(isequal(sort([pairing.BaselineRow;pairing.PostRow]),(1:84)'));
assert(isequal(string(TimeSeries.Name(pairing.BaselineRow)),pairing.BaselineLabel));
assert(isequal(string(TimeSeries.Name(pairing.PostRow)),pairing.PostLabel));
assert(isreal(TS_DataMat) && all(isfinite(TS_DataMat),'all'));

%% Put the baseline and post values into separate matrices

%Each row now refers to the same person in both matrices.
baseline = TS_DataMat(pairing.BaselineRow,:);
post = TS_DataMat(pairing.PostRow,:);
pairedDifferences = post - baseline;
numberOfFeatures = size(baseline,2);

%% Run the paired tests

%Run one signed-rank test for each feature, using all 42 participant pairs.
%Two-sided means we test for a change in either direction (higher or lower).

p = zeros(numberOfFeatures,1);

for feature = 1:numberOfFeatures
    p(feature) = signrank(post(:,feature),baseline(:,feature),'tail','both');
end

%% FDR correction

%Thousands of features are tested, so correct for multiple comparisons.
%Include ALL retained features in this PC, not just the descriptive top 40.
%Do the correction separately for PC1, PC2 and PC3.

pFDR = mafdr(p,'BHFDR',true);
significant = pFDR < fdrThreshold;

fprintf('PC%d: %d of %d features have FDR-adjusted p < %.2f.\n', ...
    pc,sum(significant),numberOfFeatures,fdrThreshold);

%% Put the results into a table

%Keep the feature ID/name so each result can be traced back to hctsa.
%Positive MedianPairedChange means the feature was higher after treatment.
%This is the median of each person's change, not a difference of group medians.

featureResults = table;
featureResults.FeatureID = Operations.ID;
featureResults.FeatureName = string(Operations.Name);
featureResults.MedianBaseline = median(baseline,1)';
featureResults.MedianPost = median(post,1)';
featureResults.MedianPairedChange = median(pairedDifferences,1)';
featureResults.P = p;
featureResults.PFDR = pFDR;
featureResults.Significant = significant;

%% Save new results, leaving previous results untouched
outputFolder = fullfile(outputRoot,sprintf('paired_PC%d',pc));
assert(~isfolder(outputFolder),'Choose a new output folder.');
mkdir(outputFolder);
writetable(featureResults,fullfile(outputFolder,'feature_results.csv'));
save(fullfile(outputFolder,'paired_results.mat'),'featureResults','pairedDifferences','pairing');
