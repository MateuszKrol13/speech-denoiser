%% Please modify before running script -> this later on will be converted 
% to a function, after i ensure that matlab fuctions can be ran on console
% with parameters
RAW_DATA_DATASET_DIR = 'cv-corpus-20.0-delta-2024-12-06\en\clips\'
DS_FORMAT = '*.mp3';

SNR_LEVEL = 4;
NOISE = 'PinkNoise';

%% CODE
recPathsStruct = dir([fullfile(pwd, '..\data\raw\', RAW_DATA_DATASET_DIR), DS_FORMAT]);
cellPaths = fullfile({recPathsStruct.folder}, {recPathsStruct.name});
audioObjContainer = cell(length(recPathsStruct), 1);

% Possibility of using multiple dataset parameters / multiple datasets,
% solution is to tie output files to datetime.
scriptLaunchDatetime = string(datetime('now', 'Format','y-MM-d_HH-mm'));
outputPath = fullfile(pwd, '..\data\interim', scriptLaunchDatetime);
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
    
    % Prepare csv metadata file, to distinguish between datasets used in
    % learning. The idea is to use a single one, but in practice this
    % varies.
    fid = fopen(fullfile(outputPath, 'METADATA.csv'), 'wt');
    fprintf(fid, ['scriptLaunchTime,%s\n', 'recordingDataDir,%s\n', 'snrLevel,%d\n', 'noiseType,%s\n'], ...
        scriptLaunchDatetime, RAW_DATA_DATASET_DIR, SNR_LEVEL, NOISE);
    fclose(fid);
end
%TODO: write a function to check against all METADATA files, to prevent
%rerunning script with the same parameters

recNumber = length(recPathsStruct);
timerStart = tic;
for recCount = 1:recNumber
    %progressbar
    if isequal(mod(recCount, floor(recNumber ./ 20)), 0)
        timer = toc(timerStart) * (recNumber - recCount) ./ recCount;
        disp(['Loaded ', num2str(recCount), ' .mp3 files out of ', num2str(recNumber), ...
            '. Estimated remaining time: ', ...
            num2str(floor(timer./60)), ' min ', ...
            num2str(mod(timer, 60)), ' s']);
    end
    
    audioObj = AudioContainer(cellPaths{recCount}, 4);
    audioObj.saveTarget(fullfile(outputPath, recPathsStruct(recCount).name));

    audioObjContainer{recCount} = audioObj;
end