% Digit Classification via a Backpropagation Network 
% Coded In:  MATLABr2013a
%
% This program uses a backpropagation artificial neural network to classify
% ASCII Hindu-Arabic numerals that were converted from handdrawn images.

% This code was written based on the YouTube playlist: Welch Labs Neural
% Networks Demystified (https://www.youtube.com/watch?v=bxe2T-V8XRs)

classdef BackPropNetwork < handle
    %BackPropNetwork: Neural network that is trained with backpropagation
    properties
        numTrainingExamples
        inputLayerSize
        hiddenLayerSize
        outputLayerSize
        learningRate
        momentum
        weightsLayer1
        weightsLayer2
    end
    
    methods
        function obj = BackPropNetwork(numTrainingExamples,inputLayerSize, hiddenLayerSize, outputLayerSize, learningRate, momentum)
            obj.numTrainingExamples = numTrainingExamples;
            obj.inputLayerSize = inputLayerSize;
            obj.hiddenLayerSize = hiddenLayerSize;
            obj.outputLayerSize = outputLayerSize;
            obj.learningRate = learningRate;
            obj.momentum = momentum;
            
            %Generate weights between 0 and 1
            obj.weightsLayer1 = -1 + (2)*rand(inputLayerSize,hiddenLayerSize);
            obj.weightsLayer2 = -1 + (2)*rand(hiddenLayerSize,outputLayerSize);
        end %End constructor
        
        function [trainedOutputs, a2, z2, z3] = forward(obj, trainingRow)
            %Calculate input to hidden nodes
            z2 = trainingRow * obj.weightsLayer1;
                    
            %Calculate hidden node activation
            [~, col] = size(z2);
            a2 = zeros(1, col);
            for i = 1:col
                a2(i) = sigmoid(z2(i)); %Use sigmoid function as activation
            end 
                    
            %Calculate input to output nodes
            z3 = a2 * obj.weightsLayer2; 
                    
            %Calculate output node activation
            [~, col] = size(z3);
            y_hat = zeros(1,col);  
            for i = 1:col
                y_hat(i) = sigmoid(z3(i)); %Use sigmoid function as activation
            end 
            trainedOutputs = y_hat;
        end
        
        function [gradientHiddenLayer, gradientInputLayer] = backpropagateError(obj, trainingOutputs,trainingLabels_vector, a2, z2, z3, trainingData)
            %First calculate the derivative of the activation function with respect to the inputs of the hidden and input nodes
            [~, col] = size(z2);
            derivativeZ2 = zeros(1, col);
            for i = 1:col
                derivativeZ2(i) = sigmoidPrime(z2(i));
            end
            
            [~, col] = size(z3);
            derivativeZ3 = zeros(1, col);
            for i = 1:col
                derivativeZ3(i) = sigmoidPrime(z3(i));
            end
            
            %Calculate the error with respect to the output layer
            deltaOutputLayer = -(trainingLabels_vector - trainingOutputs) .* derivativeZ3;
            
            %Using the error with respect to the output layer, calculate the gradient of the cost function with respect to the hidden
            %layer's weights
            gradientHiddenLayer = a2.' * deltaOutputLayer;

            %Calculate the error with respect to hidden layer
            deltaHiddenLayer = (deltaOutputLayer * obj.weightsLayer2.') .* derivativeZ2;
            
            %Using the error with respect to the output layer, calculate the gradient of the cost function with respect to the hidden
            %layer's weights
            gradientInputLayer = trainingData.' * deltaHiddenLayer;
        end
        
        function cost = train(obj, trainingData, trainingLabels_vector, epochLimit)
            %Initialize first iteration with random output
            epoch = 1;
            trainedOutputs = rand(obj.numTrainingExamples, obj.outputLayerSize);
            disp('Initial cost: ');
            initCost = obj.calcCostFxn(trainedOutputs, trainingLabels_vector);
            disp(initCost);
            cost = initCost;
            
            %Begin training through backpropagation
            while (cost > 0.05) && (epoch < epochLimit)
                for rowIndex = 1:obj.numTrainingExamples
                    %Propagate forward through the network using previous
                    %weights
                    [trainedOutputs, a2, z2, z3] = obj.forward(trainingData(rowIndex,:));
                    
                    %Backpropagate to determine the gradients of the cost
                    %function with respect to the weights in each layer
                    [gradientHiddenLayer, gradientInputLayer] = obj.backpropagateError(trainedOutputs,trainingLabels_vector(rowIndex,:), a2, z2, z3, trainingData(rowIndex,:));
                    
                    %Change weights using the backpropagated error
                    obj.weightsLayer2 = obj.weightsLayer2 - ((obj.learningRate * obj.momentum) .* gradientHiddenLayer);
                    obj.weightsLayer1 = obj.weightsLayer1 - ((obj.learningRate * obj.momentum) .* gradientInputLayer);
                    
                end %End loop for entry-by-entry training
                cost = obj.calcCostFxn(trainedOutputs, trainingLabels_vector);
                epoch = epoch + 1;
            end %End while loop for training entire network
            disp ('Final cost');
            disp(cost);
        end %End train function
        
        function [predictedLabels,accuracy] = test(obj, testingData, testingLabels)
            %Forward propagate to predict outputs using trained weights
            [numTestingExamples,~] = size(testingData);
            predictedOutputs = zeros(numTestingExamples, obj.outputLayerSize);
            
            for rowIndex = 1:numTestingExamples
                %Calculate input to hidden nodes
                [testingOutputs, ~, ~, ~] = obj.forward(testingData(rowIndex,:));
                %Append predicted testing entry to predicted outputs
                predictedOutputs(rowIndex,:) = testingOutputs;
            end
            
            numCorrect = 0;
            predictedLabels = zeros(numTestingExamples, 1);
            for rowIndex = 1:numTestingExamples
                [~, indexOfMax] = max(predictedOutputs(rowIndex,:), [], 2);
                predictedLabel = indexOfMax - 1; %Since MATLAB is 1-based indexing
                predictedLabels(rowIndex) = predictedLabel;
            end
            
            for rowIndex = 1:numTestingExamples
                if predictedLabels(rowIndex) == testingLabels(rowIndex)
                   numCorrect = numCorrect + 1;
                end
            end
            accuracy = (numCorrect/numTestingExamples)*100;
        end %End test function
        
    end %End methods
    
    methods(Static)    
        function cost = calcCostFxn(y_hat,trainingLabels_vector)
            [rowLen, numTargets] = size(y_hat);
            mseVec = zeros(rowLen, 1);
            
            %Create a vector to hold the MSE values for each predicted
            %output
            for i = 1:rowLen
                mse = 0;
                for j = 1:10
                    error = (trainingLabels_vector(i,j)-y_hat(i,j))^2;
                    mse = mse + 0.5*error;
                end
                mseVec(i) = mse;
            end %End calculating mse for one row
            
            %Take the average of the entire mseVec as the cost value
            cost = mean(mseVec);
        end %End calcCostFxn  
        
    end %End static methods
    
end %End BackPropNetwork class

