classdef AudioContainer
    %AUDIOCONTAINER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        sourceWaveform;
        targetWaveform;
        
        targetSNR;
        signalFs;
    end
    
    methods
        function obj = AudioContainer(path, snrLevel)
            %AUDIOCONTAINER Construct an instance of this class
            %   Detailed explanation goes here
            [obj.sourceWaveform, obj.signalFs] = audioread(path);
            obj.targetSNR = snrLevel;
            sourcePower = sum(obj.sourceWaveform.^2, 1)/size(obj.sourceWaveform, 1);

            noise = pinknoise(size(obj.sourceWaveform),'like',obj.sourceWaveform);
            noisePower = sum(noise.^2, 1)./size(noise, 1);

            noiseScale = sqrt(sourcePower ./ (noisePower * (10^ (obj.targetSNR/10) ) ) );
            noise =  noise.*noiseScale;

            % normalize to [-1, 1]
            obj.targetWaveform = (obj.sourceWaveform + noise) ./ max(abs(obj.sourceWaveform + noise));

        end

        function saveTarget(obj, path)
            audiowrite(path, obj.targetWaveform, obj.signalFs)
        end
    end
end

