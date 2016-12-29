% PCA Networks and the Cocktail Party Problem 
% Coded In:  MATLABr2013a
%
% This program uses a PCA network to apply PCA to find the first
% principal component of two sound sources to produce one isolated sound
% source. 

%Create a PCA network with learning rate 0.1
soundFile = csvread('sound.csv');
[numTrainingExamples,~] = size(soundFile);
learningRate = 0.1;
weightVec = [1,0];

network = PCANetwork(numTrainingExamples,learningRate,weightVec);

%Since only 1 iteration is required as per the problem guideline, there
%is no epoch limit input for the training function
network = network.train(soundFile);
trainedOutput = network.forward(soundFile);

%Write trained outputs to a .csv file
csvwrite('trainedOutputs.csv', trainedOutput);


