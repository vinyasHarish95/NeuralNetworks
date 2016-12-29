% Digit Classification via a Backpropagation Network 
% Coded In:  MATLABr2013a
%
% This program uses a backpropagation artificial neural network to classify
% ASCII Hindu-Arabic numerals that were converted from handdrawn images.

% This code was written based on the YouTube playlist: Welch Labs Neural
% Networks Demystified (https://www.youtube.com/watch?v=bxe2T-V8XRs)

% First, read the training and testing data
[trainingData,trainingLabels,testingData, testingLabels] = readData();

% Append 1 (as a bias) to the end of each row of the training data so 
% that it multiplies properly with the weights
[numExamples, ~] = size(trainingData);
biasVec = ones(numExamples,1);
trainingData = horzcat(trainingData,biasVec);

% Create an instance of the training labels in vector form
trainingLabels_vector = zeros(numExamples, 10);
for i = 1:numExamples
    target = trainingLabels(i);
    if target == 0 %Handle 0 case
        trainingLabels_vector(i) = 1;
    else
        for j = 2:10 %Fill in rest of vector
            if j == target
                trainingLabels_vector(i,j+1)=1;
            end
        end
    end
end

% Create an instance of the backpropagation network
[numTrainingExamples, inputLayerSize] = size(trainingData);
hiddenLayerSize = 16; 
outputLayerSize = 10;
learningRate = 0.01;
momentum = 1;

network = BackPropNetwork(numTrainingExamples,inputLayerSize, ... 
hiddenLayerSize, outputLayerSize, learningRate, momentum);

% Begin training network 
epochLimit = 250;
cost = network.train(trainingData,trainingLabels_vector,epochLimit);

% Test the trained network on the unseen, testing data

% Append 1 (as a bias) to the end of each row of the testing data so 
% that it multiplies properly with the weights
[numExamples, ~] = size(testingData);
biasVec = ones(numExamples,1);
testingData = horzcat(testingData,biasVec);

% Report the training accuracy
[trainingPredictedLabels, trainingAcc] = network.test(trainingData, trainingLabels);
disp('Training accuracy (%): ');
disp(trainingAcc);

% Report the testing accuracy
[testingPredictedLabels, testingAcc] = network.test(testingData, testingLabels);
disp('Testing accuracy (%): ');
disp(testingAcc);
