classdef FeatureExtractor

    methods(Static)
        function [mag, phase] = deriveSpectrogram(audioSig, varargin)
            p = inputParser;
            addRequired(p, 'Signal', @(x) isa(x, "AudioContainer"));
            addRequired(p, 'FFTLength', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'OverlapLength', @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Type', 'Symmetric', @(x) isstring(x) || ischar(x) && any(ismember({'Symmetric', 'Asymmetric'}, x)));
            addParameter(p, 'KeepPhase', false, @(x) islogical(x));
            parse(p, audioSig, varargin{:});

            spectrogram = stft(audioSig.waveform, audioSig.freq, ...
                "Window", hamming(p.Results.FFTLength, "periodic"), ...
                "OverlapLength", p.Results.OverlapLength);
            
            mag = abs(spectrogram);
            phase = angle(spectrogram);

            % discard symmetric part of magnitude
            mag = mag(1:size(mag, 1)/2 + 1, :);
        end

        function audio = reconstructAudio(spectrogram)

        end

        function resampledAudio = Resample(AudioObj, targetFs)
            src = dsp.SampleRateConverter(InputSampleRate=AudioObj.freq, ...
                OutputSampleRate=targetFs, ...
                Bandwidth=0.99.*min(AudioObj.freq, targetFs));  % for explanation see
                                                                % SampleRateConverter.m
            AudioObj.waveform = src(AudioObj.waveform);
            AudioObj.freq = targetFs;
            
            resampledAudio=AudioObj;
        end
    end
end