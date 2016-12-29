function [ trainingData,trainingLabels,testingData, testingLabels ] = readData( )
%readData reads in the training and testing data files
%   The files are split so that the data and labels are in different
%   vectors

trainingFile = csvread('training.txt');
[numTrainingEx, inputVals] = size(trainingFile);
trainingData = zeros(numTrainingEx, inputVals-1);
trainingLabels = zeros(numTrainingEx, 1);

for i = 1:numTrainingEx
    trainingData(i,:) = trainingFile(i,1:inputVals-1);
    trainingLabels(i) = trainingFile(i,inputVals);
end

testingFile = csvread('testing.txt');
[numTestingEx, inputVals] = size(testingFile);
testingData = zeros(numTestingEx, inputVals-1);
testingLabels = zeros(numTestingEx, 1);

for i = 1:numTestingEx
    testingData(i,:) = testingFile(i,1:inputVals-1);
    testingLabels(i) = testingFile(i,inputVals);
end

