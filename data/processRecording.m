function processRecording(recPath, outPath, varargin)
    % Processes an audio recording file for network inference

    p = inputParser;
    addRequired(p, 'recPath', @(x) isstring(x) || ischar(x));
    addOptional(p, 'outPath', '', @(x) ischar(x))
    addParameter(p, 'TargetFreq', 8e3, @(x) x > 0 && isinteger(x))
    addParameter(p, 'SnrLevel', 0, @(x) isinteger(x))
    parse(p, recPath, outPath, varargin{:});

    recording = AudioContainer(recPath);
    resampledAudio = FeatureExtractor.Resample(recording, p.Results.TargetFreq);
    [cleanAudio, noisyAudio] = AudioContainer.applyNoise(resampledAudio, p.Results.SnrLevel);

    [mag, phase] = FeatureExtractor.deriveSpectrogram(noisyAudio, 256, 256-64, 'Type', 'Asymmetric');
    
    for i = 1:size(mag, 2) - 8 + 1
        magData(:, :, i) = mag(:, i:i+8-1);
    end


    
    [root, ~, ~] = fileparts(recPath);
    
    if isempty(p.Results.outPath)
        outPath = fullfile(root, "inference");
        if ~exist(outPath)
            mkdir(outPath)
        end

        audiowrite(fullfile(fullfile(outPath, "cleanRec.mp3")), cleanAudio.waveform, cleanAudio.freq);
        audiowrite(fullfile(fullfile(outPath, "noisyRec.mp3")), noisyAudio.waveform, noisyAudio.freq);
        
        save(fullfile(outPath, "spectrogramMag.mat"), "magData", '-nocompression', '-v7.3');
        save(fullfile(outPath, "spectrogramAngle.mat"), "phase", '-nocompression', '-v7.3');

    else

    end

end