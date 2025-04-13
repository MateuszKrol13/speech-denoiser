classdef FeatureExtractor
    %AUDIOCONTAINER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(Constant)
        DEFAULT = 1;
    
    end

    methods(Static)
        function deriveSpectrogram(audioSig, varargin)
            p = inputParser;
            addRequired(p, 'FFTLenght', @(x) isnumeric(x) && x > 0);
            addRequired(p, 'OverlapLength', @ (x) isnumeric(x) && x > 0);
            addOptional(p, 'Type', 'Symmetric', @(x) isstring(x) && ismember(x, {'Symmetric', 'Asymmetric'}));

            [s, f ,t] = stft(audioSig.waveform, audioSig.freq, ...
                "Window", hamming(p.Results.FFTLength, "periodic"), ...
                "OverlapLength", p.Results.OverlapLength);
            s
        end

        function resampledAudio = Resample(AudioObj, targetFs)
            src = dsp.SampleRateConverter(InputSampleRate=AudioObj.freq, ...
                OutputSampleRate=targetFs, Bandwidth=min(AudioObj.freq, targetFs));
            
            AudioObj.waveform = src(AudioObj.signal);
            AudioObj.freq = targetFs;
            
            resampledAudio=AudioObj;
        end
    end
end