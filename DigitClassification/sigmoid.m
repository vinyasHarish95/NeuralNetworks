function [output] = sigmoid(x)
%sigmoid: A sigmoid function that is used as our activation function
output = 1/(1+exp(-x));
end

