% Prepare a separate hctsa copy for fresh feature extraction.
% The changes below address MATLAB compatibility and random-number handling.
% The original hctsa folder is left untouched.
% Inspect the compiler output; successful compilation alone does not establish
% that extracted feature values match a previous run.


%% File paths
% Change these paths if running on another computer.

hctsaSourceFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/from_google_drive/Manuscript/_FINAL_DRAFT_w_ALL_COAUTHOR_FEEDBACK/Neural_Networks/_Response_to_Reviewers/hctsa-main';
hctsaFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/hctsa_runtime';
supportFolder = '/Users/aron/Desktop/Neural_Networks_Paper_New_Location/GitHub_Manuscript_Code/support';



%% Make a new working copy

assert(isfolder(hctsaSourceFolder));
assert(~isfolder(hctsaFolder), 'Use a new runtime folder, not an existing installation.');

copyfile(hctsaSourceFolder, hctsaFolder);
sourceFolder = fullfile(hctsaSourceFolder, 'Operations');
targetFolder = fullfile(hctsaFolder, 'Operations');


%% Update the autocorrelation calls in the two GARCH functions

% Replace the older argument format with named arguments.
% The requested number of lags remains 20.
for name = ["MF_GARCHfit.m", "MF_GARCHcompare.m"]

    code = fileread(fullfile(sourceFolder, name));

    oldCalls = ["autocorr(y,20,[],[])", "autocorr(y.^2,20,[],[])", "parcorr(y,20,[],[])"];
    newCalls = ["autocorr(y,'NumLags',20)", "autocorr(y.^2,'NumLags',20)", "parcorr(y,'NumLags',20)"];

    for call = 1:3
        assert(count(string(code), oldCalls(call)) == 1);
        code = strrep(code, char(oldCalls(call)), char(newCalls(call)));
    end

    % Write the edited file into the working copy only.
    file = fopen(fullfile(targetFolder, name), 'w');
    assert(file ~= -1);
    fprintf(file, '%s', code);
    fclose(file);

end


%% Use scalar AR and MA orders in MF_FitSubsegments

% The operation uses orders [2,2]. Current MATLAB requires each loop limit
% to be a scalar, so specify the AR and MA entries separately. Both remain 2.
name = 'MF_FitSubsegments.m';
code = fileread(fullfile(sourceFolder, name));

start = strfind(code, "case 'arma'");
assert(numel(start) == 1);

maStart = strfind(code, '% Statistics on fitted MA parameters, q');
assert(numel(maStart) == 1);
oldLoop = 'for i = 1:order % first column will be ones';

% Separate the AR and MA sections, then update their loop limits.
pPart = code(start:maStart-1);
qPart = code(maStart:end);
assert(count(string(pPart), oldLoop) == 1 && count(string(qPart), oldLoop) == 1);

pPart = strrep(pPart, oldLoop, 'for i = 1:order(1) % explicit scalar AR order');
qPart = strrep(qPart, oldLoop, 'for i = 1:order(2) % explicit scalar MA order');
code = [code(1:start-1), pPart, qPart];

file = fopen(fullfile(targetFolder, name), 'w');
assert(file ~= -1);
fprintf(file, '%s', code);
fclose(file);
fprintf('Applied documented syntax fixes to three working-copy files.\n');


%% Reset native random generators before the TSTOOL correlation sums

% MATLAB's rng command does not reset these native generators. Insert the
% same reset before each correlation-sum calculation, regardless of labels.
name = 'NL_TSTL_GPCorrSum.m';
code = fileread(fullfile(sourceFolder, name));

marker = 'me = []; % error catcher';
assert(count(string(code), marker) == 1);

resetCode = sprintf(['%% Project reproducibility fix: reset native random state for EACH calculation.\n' ...
    '%% clear mex unloads persistent TSTOOL generators; MATLAB rng does not reset them.\n' ...
    '%% Fixed seed 20260909 is independent of participant, timepoint and class labels.\n' ...
    'clear mex;\nhctsa_seed_c_rng(20260909);\n']);

code = strrep(code, marker, [resetCode, marker]);

file = fopen(fullfile(targetFolder, name), 'w');
assert(file ~= -1);
fprintf(file, '%s', code);
fclose(file);


%% Compile the native routines using hctsa's build script

previousFolder = pwd;
cd(hctsaFolder);
clear startup;
startup;

cd(fullfile(hctsaFolder, 'Toolboxes'));
compile_mex;

% Compile the small helper used to set the native random seed.
mex('-outdir', supportFolder, fullfile(supportFolder, 'hctsa_seed_c_rng.c'));

cd(previousFolder);
