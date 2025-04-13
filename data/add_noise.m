%% Script options
RAW_DATA_DATASET_DIR = 'cv-corpus-20.0-delta-2024-12-06\en\clips\'
DS_FORMAT = '*.mp3';

SNR_LEVEL = 0;
NOISE = 'PinkNoise';

TRAIN_SPLIT = 0.9;
TEST_SPLIT = 0.05;
VAL_SPLIT = 0.05;

%% Notes
%TODO: write a function to check against all METADATA files, to prevent
%rerunning script with the same parameters

%% CODE
% ####################################################################### %
% ################## Load recordings and apply noise #################### %
% ####################################################################### %
recNumber = 100;
recPathsStruct = dir([fullfile(pwd, 'raw', RAW_DATA_DATASET_DIR), DS_FORMAT]);
cellPaths = fullfile({recPathsStruct.folder}, {recPathsStruct.name});
%audioObjContainer = cell(length(recPathsStruct), 2);
audioObjContainer = cell(recNumber, 2);

timerStart = tic;
fprintf("Loading %d recordings...", recNumber);
for recCount = 1:recNumber
    %progressbar
    if isequal(mod(recCount, floor(recNumber ./ 20)), 0)
        timer = toc(timerStart) * (recNumber - recCount) ./ recCount;
        disp(['Loaded ', num2str(recCount), ' .mp3 files out of ', num2str(recNumber), ...
            '. Estimated remaining time: ', ...
            num2str(floor(timer./60)), ' min ', ...
            num2str(mod(timer, 60)), ' s']);
    end
    
    [cleanAudio, noisyAudio] = AudioContainer.applyNoise(AudioContainer(cellPaths{recCount}), SNR_LEVEL);
    [audioObjContainer{recCount, :}] = deal(cleanAudio, noisyAudio);
end

% ####################################################################### %
% ################# Split into TEST / TRAIN / VALIDATE ################## %
% ####################################################################### %
totalIdx = 1:size(audioObjContainer, 1);

trainIdx = ismember(totalIdx, randperm(recNumber, TRAIN_SPLIT.*recNumber));
trainRecs = audioObjContainer(trainIdx, :);
testValRecs = audioObjContainer(~(trainIdx), :);

testValIdx = 1:size(testValRecs, 1);
valTestSplit = VAL_SPLIT ./ (VAL_SPLIT + TEST_SPLIT);

valIdx = ismember(testValIdx, ...
    randperm(length(testValIdx), floor(length(testValIdx) .* valTestSplit ) ));
valRecs = testValRecs(valIdx, :);
testRecs = testValRecs(~(valIdx), :);


% ####################################################################### %
% ######################## create file structure ######################## %
% ####################################################################### %

scriptLaunchDatetime = string(datetime('now', 'Format','y-MM-d_HH-mm'));
interimPath = fullfile(pwd, 'interim', scriptLaunchDatetime);
processedPath = fullfile(pwd, 'processed', scriptLaunchDatetime);

if ~exist(processedPath, 'dir')
    mkdir(processedPath);

    % Handle train data
    mkdir(fullfile(processedPath, 'train'));
    % HERE SAVE THE SPECTROGRAMS AS .mat

    % Handle validate data
    mkdir(fullfile(processedPath, 'validate'));
    % HERE SAVE THE SPECTROGRAMS AS .mat

    % Handle test data
    testPath = fullfile(processedPath, 'test');
    mkdir(testPath);
    mkdir(fullfile(testPath, 'cleanrecs'));
    mkdir(fullfile(testPath, 'noisyrecs'));
    % HERE SAVE THE SPECTROGRAMS AS .mat
    cellfun(@(rec) rec.saveTarget(fullfile(testPath, 'cleanrecs')), testRecs(:, 1));
    cellfun(@(rec) rec.saveTarget(fullfile(testPath, 'noisyrecs')), testRecs(:, 2));
    
    % Prepare csv metadata file, to distinguish between datasets used in
    % learning. The idea is to use a single one, but in practice this
    % varies.
    fid = fopen(fullfile(outputPath, 'METADATA.csv'), 'wt');
    fprintf(fid, ['scriptLaunchTime,%s\n', 'recordingDataDir,%s\n', 'snrLevel,%d\n', 'noiseType,%s\n'], ...
        scriptLaunchDatetime, RAW_DATA_DATASET_DIR, SNR_LEVEL, NOISE);
    fclose(fid);
end


