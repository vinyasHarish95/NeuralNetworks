function [output] = sigmoid_prime(x)
%sigmoid prime: The derivative of the sigmoid function that is used as 
%our activation function
output = exp(x)/((exp(x)+1)^2);
end

