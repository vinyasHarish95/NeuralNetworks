PCA Networks and the Cocktail Party Problem
Coded In:  MATLABr2013a

This program uses a PCA network to apply PCA to find the first
principal component of two sound sources to produce one isolated sound
source. I set no epoch limit for the training function, which ended up being fine as an adequate solution was determined in one iteration.  The variable 'k' was based on
William's suggestion (1985) and was scaled by the learning rate to avoid
'NaN' errors.  Using the outputs with initial weights and the 'k' value,
the weight change was calculated and applied.  Following the weight change,
the inputs were passed through the network again to generate trained outputs--
which are saved in the 'trainedOutputs.csv' file and as a sound in the
'outputSound.wav' file.  

The final weights were as follows:

                            0.2239   -0.2488
