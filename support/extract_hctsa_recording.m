function result = extract_hctsa_recording(inputSignal, template, Operations, MasterOperations, scratchFile, seed)
% Calculate hctsa features for one recording's PC signal.
%
% Called by C_extract_fold_features.m. inputSignal has 4,800 samples;
% template holds the recording's metadata. The operation tables specify
% the original hctsa features and their parameters.
%
% scratchFile is a temporary HCTSA file in the new output folder.
% This function calls TS_Compute; it does not replace hctsa's algorithms.
% hctsa credit/licences: see the preserved runtime_hctsa source distribution.


%% Check the signal and feature definitions

assert(numel(inputSignal) == 4800 && all(isfinite(inputSignal)));
assert(height(template) == 1 && height(Operations) == 7479);


%% Put the signal into the file format hctsa expects

TimeSeries = template;
TimeSeries.Data{1} = inputSignal(:);

% hctsa fills these arrays with feature values, quality flags and timings.
TS_DataMat = nan(1, height(Operations));
TS_Quality = TS_DataMat;
TS_CalcTime = TS_DataMat;
fromDatabase = false;

save(scratchFile, 'TimeSeries', 'Operations', 'MasterOperations', ...
    'TS_DataMat', 'TS_Quality', 'TS_CalcTime', 'fromDatabase', '-v7.3');


%% Reset the random generators before calculating features

% Reset both MATLAB's random stream and the native stream used by some
% compiled hctsa routines. Use the same seed for every signal.
rng(seed, 'twister');
clear corrsum corrsum2;
hctsa_seed_c_rng(seed);


%% Run hctsa and read its outputs

timer = tic;
TS_Compute(false, TimeSeries.ID, [], 'missing', scratchFile, 'minimal');

result = load(scratchFile, 'TS_DataMat', 'TS_Quality', 'TS_CalcTime', ...
    'TimeSeries', 'Operations', 'MasterOperations');
result.wallSeconds = toc(timer);


%% Check that hctsa retained the recording and feature identities

assert(isequaln(result.Operations, Operations) ...
    && isequaln(result.MasterOperations, MasterOperations));
assert(isequaln(result.TimeSeries, TimeSeries), 'hctsa changed input identity.');
assert(all(isfinite(result.TS_Quality), 'all'), 'Some features were not attempted.');

% Keep failed calculations and their quality flags too. Feature filtering
% happens later, using the training recordings for each fold.

end
