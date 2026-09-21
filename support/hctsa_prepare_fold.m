function [prepared, featureColumns, replacements] = hctsa_prepare_fold(rawFeatures, quality, isTraining)
% Prepare the hctsa features for one training/test split.
%
% rawFeatures contains recordings in rows and features in columns.
% quality contains hctsa's quality flags; zero means the calculation passed.
% isTraining identifies the recordings used to prepare the features.
%
% The output contains the retained, normalised features for all recordings.
% featureColumns identifies their original columns; replacements records
% any invalid test values replaced with a training median.
%
% Credit: hctsa, Fulcher et al. (2013), Fulcher & Jones (2017), GPL-3.0.


%% Check the input sizes and training rows

assert(isnumeric(rawFeatures) && ismatrix(rawFeatures));
assert(isequal(size(rawFeatures), size(quality)), 'Quality/value sizes differ.');

assert(islogical(isTraining) && isvector(isTraining) ...
    && numel(isTraining) == size(rawFeatures, 1), ...
    'One logical per recording required.');

isTraining = isTraining(:);
assert(sum(isTraining) >= 2, 'At least two training recordings required.');

% Values must be finite, real and have a successful hctsa quality flag.
valid = isfinite(rawFeatures) & (quality == 0) & (imag(rawFeatures) == 0);

% Check the original 80% recording-quality threshold. Stop if a training
% recording fails, rather than dropping it and breaking a participant pair.
assert(all(mean(valid(isTraining, :), 2) >= 0.8), ...
    'A training recording fails the original 80%% quality rule. Investigate.');


%% Keep features that can be used in the training data

% A feature must have valid values in every training recording.
qualityColumns = find(all(valid(isTraining, :), 1));
assert(~isempty(qualityColumns), 'No features pass training quality checks.');

% Also exclude features with almost no variation in the training data.
% The normalisation helper checks this before and after scaling.
[~, trainingKeep] = hctsa_normalise_training( ...
    real(rawFeatures(isTraining, qualityColumns)), true(sum(isTraining), 1));

featureColumns = qualityColumns(trainingKeep);


%% Replace invalid test values with the training median

% Training values are not replaced. Each test value needing replacement
% receives the median of that feature's raw training values.
modelRaw = real(rawFeatures(:, featureColumns));
trainingMedians = median(modelRaw(isTraining, :), 1);

badTest = ~valid(:, featureColumns) & ~isTraining;
[recordingRows, localColumns] = find(badTest);

% Record which values were replaced, using the original feature columns.
inputColumns = reshape(featureColumns(localColumns), [], 1);
originalIndex = sub2ind(size(rawFeatures), recordingRows, inputColumns);
replacementRaw = reshape(trainingMedians(localColumns), [], 1);

replacements = table( ...
    recordingRows, inputColumns, rawFeatures(originalIndex), ...
    quality(originalIndex), replacementRaw, ...
    'VariableNames', ...
    {'RecordingRow', 'RawFeatureColumn', 'OriginalStoredValue', ...
    'QualityFlag', 'TrainingMedianRaw'});

modelRaw(badTest) = replacementRaw;
assert(all(isfinite(modelRaw), 'all'), 'Prepared raw input still invalid.');


%% Normalise using training data, then apply the same transformation to test data

[prepared, keepAgain] = hctsa_normalise_training(modelRaw, isTraining);

assert(all(keepAgain), 'Feature eligibility changed unexpectedly.');
assert(all(isfinite(prepared), 'all'));

% Test values can fall outside the training-derived 0–1 range.
% Leave them as calculated; do not clip them to that range.

end
