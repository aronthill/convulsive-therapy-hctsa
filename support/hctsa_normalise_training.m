function [normalised, keepFeatures] = hctsa_normalise_training(rawFeatures, isTraining)
% Normalise features using hctsa's mixed-sigmoid approach.
%
% All choices and scaling parameters are based on training recordings only.
% The same transformation is then applied to training and test recordings.
% keepFeatures identifies usable columns; excluded columns are left as NaN.
%
% BF_NormalizeMatrix supplies the hctsa transformations. This helper chooses
% which transformation to use from the training data, because the supplied
% hctsa mixedSigmoid option makes that choice using the full input matrix.
%
% hctsa credit: Fulcher et al. (2013), Fulcher & Jones (2017), GPL-3.0.


%% Check the inputs

assert(isnumeric(rawFeatures) && ismatrix(rawFeatures) && isreal(rawFeatures) ...
    && all(isfinite(rawFeatures), 'all'), 'Expected finite real feature values.');

assert(islogical(isTraining) && isvector(isTraining) ...
    && numel(isTraining) == size(rawFeatures, 1), ...
    'Provide one logical per recording.');

isTraining = isTraining(:);
assert(sum(isTraining) >= 2, 'At least two training recordings are required.');

trainingValues = rawFeatures(isTraining, :);


%% Exclude features with almost no variation in the training data

% Use the same near-constant threshold as hctsa's TS_Normalize.
keepFeatures = std(trainingValues, 0, 1) >= 10*eps;
normalised = NaN(size(rawFeatures));


%% Choose the appropriate sigmoid transformation

% hctsa uses an ordinary sigmoid when the interquartile range is zero;
% otherwise it uses the robust sigmoid. Make this choice on training rows.
useOrdinary = keepFeatures & (iqr(trainingValues, 1) == 0);
useRobust = keepFeatures & ~useOrdinary;

% Passing isTraining tells hctsa which rows to use to estimate the scaling.
normalised(:, useOrdinary) = BF_NormalizeMatrix( ...
    rawFeatures(:, useOrdinary), 'scaledSigmoid', isTraining);

normalised(:, useRobust) = BF_NormalizeMatrix( ...
    rawFeatures(:, useRobust), 'scaledRobustSigmoid', isTraining);


%% Check the features again after normalisation

% Retain features that are finite and still vary in the training data.
keepFeatures = keepFeatures & all(isfinite(normalised(isTraining, :)), 1) ...
    & (std(normalised(isTraining, :), 0, 1) >= 10*eps);

normalised(:, ~keepFeatures) = NaN;

assert(any(keepFeatures), 'No usable training features remain.');
assert(all(isfinite(normalised(:, keepFeatures)), 'all'), ...
    'A retained feature could not be transformed: inspect it before modelling.');

% Test values may fall outside 0–1 after training-derived scaling.
% This is expected when applying the same transformation; do not clip them.

end
