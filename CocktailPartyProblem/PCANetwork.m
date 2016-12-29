% CISC452 Assignment 3: PCA Networks and the Cocktail Party Problem 
% Coded By: Vinyas Harish 12vh6 10089169
% Coded In:  MATLABr2013a
%
% This program uses a PCA network to apply PCA to find the first
% principal component of two sound sources to produce one isolated sound
% source. 

classdef PCANetwork < handle
    properties
        weightVec
    end %End public properties
    
    properties (Access = private)
        numTrainingExamples
        learningRate
    end %End private properties
    
    methods
        function obj = PCANetwork(numTrainingExamples,learningRate,weightVec)
            obj.numTrainingExamples = numTrainingExamples;
            obj.learningRate = learningRate;
            obj.weightVec = weightVec;
        end %End constructor
        
        function obj = train(obj, trainingData)
            %Calculate activity of PCA network nodes
            activityVec = zeros(obj.numTrainingExamples, 1);
            for i = 1:obj.numTrainingExamples
                activation = trainingData(i, :).*obj.weightVec;
                activity = sum(activation);
                activityVec(i) = activity;
            end %End for loop 
                
            %Calculate k based on William's suggestion (1985) and scale
            %it by the learning rate
            k = (activityVec' * activityVec) * obj.learningRate;
              
            %Determine the weight change, delta W 
            for i = 1:obj.numTrainingExamples
                deltaW = (obj.learningRate .* activityVec(i)) * (trainingData(i, :) - k .* obj.weightVec);
                obj.weightVec = obj.weightVec + deltaW;
            end %End loop to calculate the outputs
            
            disp('Final weights:');
            disp(obj.weightVec);
        end %End train function
        
        function trainedOutput = forward(obj, trainingData)
            %Calculate the outputs (y)
            trainedOutput = zeros(obj.numTrainingExamples,1);
            for i = 1:obj.numTrainingExamples
                %Calculate activity of PCA network nodes
                activation = trainingData(i, :).*obj.weightVec;
                activity = sum(activation);
                trainedOutput(i) = activity;
            end %End for loop 
            
            %Write the trained output to a .wav file
            audiowrite('outputSound.wav', trainedOutput,8192);
        end %End feedforward function
                    
    end %End methods 
    
end %End PCANetwork class

