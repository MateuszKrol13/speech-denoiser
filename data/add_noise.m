%% Script options
RAW_DATA_DATASET_DIR = 'cv-corpus-20.0-delta-2024-12-06\en\clips\'
DS_FORMAT = '*.mp3';

SNR_LEVEL = 0;
TARGET_FREQ = 8e3;
NOISE = 'PinkNoise';
OUTPUT_DATA = 'Spectrogram absolute values'


TRAIN_SPLIT = 0.95;
TEST_SPLIT = 1 - TRAIN_SPLIT;



REC_NUMBER = 500;

%% Notes
%TODO: write a function to check against all METADATA files, to prevent
%rerunning script with the same parameters

%% CODE
% ####################################################################### %
% ################## Load recordings and apply noise #################### %
% ####################################################################### %

% pick recordings randomly
recPathsStruct = dir([fullfile(pwd, 'raw', RAW_DATA_DATASET_DIR), DS_FORMAT]);
cellPaths = fullfile({recPathsStruct.folder}, {recPathsStruct.name});
cellPaths = cellPaths(randperm(length(cellPaths))); % shuffle recordings

audioObjContainer = cell(REC_NUMBER, 2);
spectrogramContainer = cell(REC_NUMBER, 2);
phaseContainer = cell(REC_NUMBER, 2);

timerStart = tic;
fprintf("Loading %d recordings...", REC_NUMBER);
for recCount = 1:REC_NUMBER
    %progressbar
    if isequal(mod(recCount, floor(REC_NUMBER ./ 50)), 0)
        timer = toc(timerStart) * (REC_NUMBER - recCount) ./ recCount;
        disp(['Loaded ', num2str(recCount), ' .mp3 files out of ', num2str(REC_NUMBER), ...
            '. Estimated remaining time: ', ...
            num2str(floor(timer./60)), ' min ', ...
            num2str(mod(timer, 60)), ' s']);
    end

    % get audio
    sourceSampleAudio = AudioContainer(cellPaths{recCount});
    resampledAudio = FeatureExtractor.Resample(sourceSampleAudio, TARGET_FREQ);
    [cleanAudio, noisyAudio] = AudioContainer.applyNoise(resampledAudio, SNR_LEVEL);
    [audioObjContainer{recCount, :}] = deal(cleanAudio, noisyAudio);

    % get spectrogram
    cleanSpectrogram = FeatureExtractor.deriveSpectrogram(cleanAudio, 256, 256-64, 'Type', 'Asymmetric');
    noisySpectrogram = FeatureExtractor.deriveSpectrogram(noisyAudio, 256, 256-64, 'Type', 'Asymmetric');
    noisySequences = zeros(size(noisySpectrogram, 1), 8, size(noisySpectrogram, 2) - 8 + 1, 'double');
    
    for i = 1:size(noisySpectrogram, 2) - 8 + 1
        noisySequences(:, :, i) = noisySpectrogram(:, i:i+8-1);
    end
    cleanSequences = cleanSpectrogram(:, 8:end);
    [spectrogramContainer{recCount, :}] = deal(cleanSequences, noisySequences);
end

% ####################################################################### %
% ################# Split into TEST / TRAIN / VALIDATE ################## %
% ####################################################################### %
totalIdx = 1:size(audioObjContainer, 1);

% Dataset split
trainIdx = ismember(totalIdx, randperm(REC_NUMBER, TRAIN_SPLIT.*REC_NUMBER));
trainFeaturesCell = spectrogramContainer(trainIdx, :);
testFeaturesCell = spectrogramContainer(~(trainIdx), :);
testRecordingsCell = audioObjContainer(~(trainIdx), :);

% Train data
sourceTrain = cat(3, trainFeaturesCell{:, 2});
targetTrain = cat(2, trainFeaturesCell{:, 1});

% Test data
sourceTest = cat(3, testFeaturesCell{:, 2});
targetTest = cat(2, testFeaturesCell{:, 1});

% ####################################################################### %
% ######################## create file structure ######################## %
% ####################################################################### %

scriptLaunchDatetime = string(datetime('now', 'Format','y-MM-d_HH-mm'));

%paths
processedPath = fullfile(pwd, 'processed', scriptLaunchDatetime);
trainPath = fullfile(processedPath, 'train');
testPath = fullfile(processedPath, 'test');

if ~exist(processedPath, 'dir')
    mkdir(processedPath);

    % Handle train data
    mkdir(trainPath);
    save(fullfile(trainPath, "source.mat"), "sourceTrain",'-nocompression', '-v7.3');
    save(fullfile(trainPath, "target.mat"), "targetTrain", '-nocompression', '-v7.3');

    % Handle test data
    mkdir(testPath);
    save(fullfile(testPath, "source.mat"), "sourceTest", '-nocompression', '-v7.3');
    save(fullfile(testPath, "target.mat"), "targetTest", '-nocompression', '-v7.3');

    mkdir(fullfile(testPath, 'cleanrecs'));
    mkdir(fullfile(testPath, 'noisyrecs'));
    cellfun(@(rec) rec.saveTarget(fullfile(testPath, 'cleanrecs')), testRecordingsCell(:, 1));
    cellfun(@(rec) rec.saveTarget(fullfile(testPath, 'noisyrecs')), testRecordingsCell(:, 2));
    
    % Prepare csv metadata file, to distinguish between datasets used in
    % learning. The idea is to use a single one, but in practice this
    % varies.
    fid = fopen(fullfile(processedPath, 'METADATA.csv'), 'wt');
    fprintf(fid, ['scriptLaunchTime,%s\n', 'recordingDataDir,%s\n', 'snrLevel,%d\n', 'noiseType,%s\n'], ...
        scriptLaunchDatetime, RAW_DATA_DATASET_DIR, SNR_LEVEL, NOISE);
    fclose(fid);
end


