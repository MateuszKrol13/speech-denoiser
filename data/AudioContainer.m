classdef AudioContainer
    %AUDIOCONTAINER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        waveform
        freq
        recName

        targetSNR
    end
    
    methods
        function obj = AudioContainer(path)
            %AUDIOCONTAINER Construct an instance of this class
            %   Detailed explanation goes here
            if nargin < 1
                obj.waveform = double.empty;
                obj.freq = double.empty;
                obj.recName = string.empty;
            else
                [obj.waveform, obj.freq] = audioread(path);
                [~, fileName, ext] = fileparts(path);
                obj.recName = strcat(fileName, ext);
            end

            obj.targetSNR = double.empty;
        end

        function saveTarget(obj, path)
            audiowrite(fullfile(path, obj.recName), obj.waveform, obj.freq)
        end
    end

    methods(Static)
        
        function [audioObj, noisySignal] = applyNoise(audioObj, snrLevel)
            sourcePower = sum(audioObj.waveform.^2, 1)/size(audioObj.waveform, 1);

            % Create pink noise with power level corresponding to target SNR
            noise = pinknoise(size(audioObj.waveform),'like', audioObj.waveform);
            noisePower = sum(noise.^2, 1) ./size(noise, 1);
            noise =  noise.* sqrt(sourcePower ./ (noisePower * (10^ (snrLevel /10) )));

            % copy, apply noise and normalize to [-1, 1]
            noisySignal = audioObj;
            noisySignal.waveform = (noisySignal.waveform + noise) ./ max(abs(noisySignal.waveform + noise));
            noisySignal.targetSNR = snrLevel;
        end

    end
end

