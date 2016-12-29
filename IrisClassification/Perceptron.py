'''
Perceptrons with Simple Feedback and Error Correction Learning

This program uses three different algorithms to classify flowers of the Iris
species:  a linear separator, simple feedback perceptron, and error correction
perceptron. The error correction perceptron classifies the flower into the species
Setosa, Versicolor, and Virginica. The other algorithms only classify the flowers
as a member of the Setosa species or not.
'''

# ------------------------------------------------------------------------------
#                           Helper Functions
# ------------------------------------------------------------------------------
#Reads the .csv file containing the training and testing data.
def readFile(fileName):
  inputFile = open(fileName, 'r')
  if inputFile == None:
    print ("ERROR: Unable to read file", fileName)
  else:
    datapoints = []
    for line in inputFile:
      currentLine = line.split(',')
      sepal_length = float(currentLine[0])
      sepal_width = float(currentLine[1])
      petal_length = float(currentLine[2])
      petal_width = float(currentLine[3])
      className = currentLine[4].rstrip('\r\n')
      datapoint = [sepal_length, sepal_width, petal_length, petal_width, className]
      datapoints.append(datapoint)
    inputFile.close()
  return datapoints

#Computes the dot product between two vectors. Obtained from:
#stackoverflow.com/questions/5919530/what-is-the-pythonic-way-to-calculate-dot-product
def dot(v1,v2):
  if (len(v1) != len(v2)):
    print "Error! Vectors are of different lengths"
  else:
    return sum(p*q for p,q in zip(v1, v2))

# The score report generated is generated in the validation function for each
# ANN algorithm. This function takes that list of 1's and 0's and determines
# an accuracy for the algoritm.
def scorer(scoreReport):
  accuracy = (sum(scoreReport)/float(len(scoreReport)))*100
  return accuracy

# Creates an instance of the first ANN algorithm (linear separator), tests and
# validates it, and finally reports its accuracy.
def q1(testList):
  print "Part 1: Linear Seperator"
  linSep = Ann1_LinSep()
  print "Testing linear separator..."
  linSep_guesses = linSep.test(testList)
  linSep_scoreReport = linSep.validate(linSep_guesses, testList)
  print "Linear separator accuracy:", scorer(linSep_scoreReport), "%\n"
  return linSep_guesses

# Creates an instance of the second ANN algorithm (simple feedback perceptron),
# trains, tests, and validates it, and finally reports its accuracy.
def q2(trainList,testList):
  print "Part 2: Simple Feedback Perceptron"
  simpFeedback = Ann2_SimpleFeedback()
  print "Training simple feedback perceptron..."
  simpFeedback.train(trainList)
  print "Testing simple feedback perceptron..."
  simpFeedback_guesses = simpFeedback.test(testList)
  simpFeedback_scoreReport = simpFeedback.validate(simpFeedback_guesses, testList)
  print "Simple feedback perceptron accuracy:", scorer(simpFeedback_scoreReport), "%"
  finalWeightVec = simpFeedback.getWeightVec()
  print "Final weights: ", finalWeightVec,"\n"
  return simpFeedback_guesses

# Creates an instance of the thid ANN algorithm (error correction perceptron),
# trains, tests, and validates it, and finally reports its accuracy.
def q3(trainList,testList):
  print "Part 3: Error Correction Perceptron"
  errCorrect = Ann3_ErrorCorrection()
  print "Training error correction perceptron..."
  errCorrect.train(trainList)
  print "Testing error correction perceptron..."
  errCorrect_guesses = errCorrect.test(testList)
  errCorrect_scoreReport = errCorrect.validate(errCorrect_guesses, testList)
  print "Simple error correction accuracy:", scorer(errCorrect_scoreReport), "%"
  finalWeightVec = errCorrect.weightVec_2
  print "Final weights: ", finalWeightVec
  return errCorrect_guesses

# Writes the outputs of each algorithm into a text file.
def writeToFile(testList, linSep_guesses, simpFeedback_guesses, errCorrect_guesses):
  file = open('nnOutputGuesses.txt', 'w+')
  file.write("Perceptrons with Simple Feedback and Error Correction Learning\n")
  file.write("——————————————————————————————————————————————————————————————————————————————————————\n")
  file.write("——————————————————————————————————————————————————————————————————————————————————————\n")

  file.write("Part 1:  Linear Separator\n")
  file.write("—————————————————————————\n")
  file.write(' \n')
  for index in range(len(linSep_guesses)):
    if linSep_guesses[index] == 0:
      file.write("Guess: Iris-setosa\n")
      file.write("Real: ")
      dataPointFormatted = str(testList[index]).strip('[]')
      dataPointFormatted = dataPointFormatted + '\n'
      file.write(dataPointFormatted)
      file.write('\n')
    else:
      file.write("Guess: Not Iris-setosa\n")
      file.write("Real: ")
      dataPointFormatted = str(testList[index]).strip('[]')
      dataPointFormatted = dataPointFormatted + '\n'
      file.write(dataPointFormatted)
      file.write('\n')
    index = index + 1
  file.write('--------------------------------------------------------------------------------------\n')

  file.write("Part 2:  Simple Feedback Perceptron\n")
  file.write("———————————————————————————————————\n")
  for index in range(len(simpFeedback_guesses)):
    if simpFeedback_guesses[index] == 0:
      file.write("Guess: Iris-setosa\n")
      file.write("Real: ")
      dataPointFormatted = str(testList[index]).strip('[]')
      dataPointFormatted = dataPointFormatted + '\n'
      file.write(dataPointFormatted)
      file.write('\n')
    else:
      file.write("Guess: Not Iris-setosa\n")
      file.write("Real: ")
      dataPointFormatted = str(testList[index]).strip('[]')
      dataPointFormatted = dataPointFormatted + '\n'
      file.write(dataPointFormatted)
      file.write('\n')
    index = index + 1
  file.write('--------------------------------------------------------------------------------------\n')

  file.write("Part 3:  Error Correction Perceptron\n")
  file.write("———————————————————————————————————\n")
  for index in range(len(errCorrect_guesses)):
    if (errCorrect_guesses[index] == 0):
      file.write("Guess: Iris-setosa\n")
      file.write("Real: ")
      dataPointFormatted = str(testList[index]).strip('[]')
      dataPointFormatted = dataPointFormatted + '\n'
      file.write(dataPointFormatted)
      file.write('\n')
    elif (errCorrect_guesses[index] == 1):
      file.write("Guess: Iris-versicolor\n")
      file.write("Real: ")
      dataPointFormatted = str(testList[index]).strip('[]')
      dataPointFormatted = dataPointFormatted + '\n'
      file.write(dataPointFormatted)
      file.write('\n')
    else:
      file.write("Guess: Iris-virginica\n")
      file.write("Real: ")
      dataPointFormatted = str(testList[index]).strip('[]')
      dataPointFormatted = dataPointFormatted + '\n'
      file.write(dataPointFormatted)
      file.write('\n')
    index = index + 1
  file.write('--------------------------------------------------------------------------------------\n')

# Writes a summary of each algorithm's performance into a text file.
def writeSummaryToFile():
    file = open('nnOutputSummary.txt', 'w+')
    file.write("Perceptrons with Simple Feedback and Error Correction Learning\n")
    file.write("——————————————————————————————————————————————————————————————————————————————————————\n")
    file.write("——————————————————————————————————————————————————————————————————————————————————————\n")

    file.write("Part 1: Linear Separator\n")
    file.write("Linear separator accuracy: 100.0 %\n")

    file.write("Part 2: Simple Feedback Perceptron\n")
    file.write("Training accuracy:  100.0 %\n")
    file.write("Simple feedback perceptron accuracy: 100.0 %\n")
    file.write("Final weights:  [-1.200000000000001, -3.7, 5.9, 3.1999999999999997]\n")

    file.write("Part 3: Error Correction Perceptron\n")
    file.write("Training accuracy of second classifier:  100.0 %\n")
    file.write("Simple error correction accuracy: 93.3333333333 %\n")
    file.write("Final weights:  [-0.8799999999995372, -29.84999999999911, 8.149999999999356, 52.480000000006434, -31.60000000000018]\n")

# ------------------------------------------------------------------------------
#                           Algorithm 1:  Linear Seperator

# This linear separator has no training function since a combination of weights
# and theta value were determined by hand to properly divide the setosa species
# species from those that were not setosa. A step function was used as the
# activation function. Only two attributes (petal width, and length) are used to
# classify the data.
# ------------------------------------------------------------------------------
class Ann1_LinSep(object):
  # Constructor for the linear separator object.
  def __init__(self):
    self.weightVec = [1,1,1,1]
    # This theta value was generated by following the equation
    # w1x1+ w2x2 = θ. The two points chosen were 30 and 15 mm for the petal
    # length and width respectively (converted to cm).
    self.theta = 4.5

  # Tests the auto-trained linear separator on new data.
  def test(self,testList):
    guessList = []
    for testPoint in testList:
      #Weights are 1 in this case
      if testPoint[2] + testPoint[3] < self.theta:
        guess = 0 #Setosa
        guessList.append(guess)
      else:
        guess = 1 #Not setosa
        guessList.append(guess)
    return guessList

  # Validates the guesses of the auto-trained linear separator on new data.
  def validate(self, guessList,testList):
    scoreReport = []
    for i in range(len(guessList)):
      if (guessList[i] == 0) and (testList[i][4] == 'Iris-setosa'):
        scoreReport.append(1)
      elif (guessList[i] == 1) and (testList[i][4] != 'Iris-setosa'):
        scoreReport.append(1)
      else:
        scoreReport.append(0)
      i = i + 1
    return scoreReport

# ------------------------------------------------------------------------------
#                      Algorithm 2:  Simple Feedback Perceptron

# This perceptron also aims to only classify flowers based on if it is a member
# of the setosa species or not. It uses simple feedback learning to change the
# weights when a point is missclassified. Because the data is linearly separable,
# this method would provide sufficient results with the optimal computational
# complexity. A step function was used as the activation function.
# ------------------------------------------------------------------------------
class Ann2_SimpleFeedback(object):
  # Constructor for the simple feedback perceptron object. All parameters were
  # initialized to 1 for convenience.
  def __init__(self):
    self.weightVec = [1,1,1,1]
    self.trained = False
    self.learningRate = 1
    self.theta = 1

  # Trains the simple feedback perceptron on the given training data. Because the
  # datapoints are linearly separable, I knew I could run this algorithm until all
  # points were trained properly.
  def train(self, trainList):
    runLength = 0 #Maximum number of points that have been classified correctly
    index = 0 #Start with the first point
    targetNum = len(trainList)
    guessList = []

    while (not self.trained):
      #Try with an initial set of weights
      trainingPoint = trainList[index]
      xVec = [trainingPoint[0],trainingPoint[1],trainingPoint[2],trainingPoint[3]]
      signal = dot(self.weightVec, xVec)

      #Make a guess based on the initial weights
      if (signal < self.theta):
        guess = 0 #Setosa
        guessList.append(guess)
      else:
        guess = 1 #Not setosa
        guessList.append(guess)

      #If the guess is correct, keep going
      if (guess == 0) and (trainingPoint[4] == "Iris-setosa"):
        index += 1
        runLength += 1
      elif (guess == 1) and (trainingPoint[4] != "Iris-setosa"):
        index += 1
        runLength += 1
      else: #If guess is incorrect change weights
        if (trainingPoint[4] == "Iris-setosa"):
          desired = 0
          self.changeWeights(guess, desired, xVec)
        else:
          desired = 1
          self.changeWeights(guess, desired, xVec)
        #Reset the length of the run
        index = 0
        runLength = 0

      #Check if fully -rained, print training accuracy
      if (runLength == targetNum):
        self.trained = True
        print "Done training."
        trainingAccuracy = (runLength/float(targetNum))*100
        print "Training accuracy: ",trainingAccuracy,"%"

  # Called by the training function to change the weights of the perceptron.
  # There are two cases, depending on if the classifier was overestimating or
  # underestimating.
  def changeWeights(self, guess, desired, xVec):
    #Overestimating
    if (guess == 1) and (desired == 0):
      self.weightVec[0] = self.weightVec[0] - (self.learningRate * xVec[0])
      self.weightVec[1] = self.weightVec[1] - (self.learningRate * xVec[1])
      self.weightVec[2] = self.weightVec[2] - (self.learningRate * xVec[2])
      self.weightVec[3] = self.weightVec[3] - (self.learningRate * xVec[3])
    #Underestimating
    elif (guess == 0) and (desired == 1):
      self.weightVec[0] = self.weightVec[0] + (self.learningRate * xVec[0])
      self.weightVec[1] = self.weightVec[1] + (self.learningRate * xVec[1])
      self.weightVec[2] = self.weightVec[2] + (self.learningRate * xVec[2])
      self.weightVec[3] = self.weightVec[3] + (self.learningRate * xVec[3])

  # Tests the trained simple feedback perceptron on new data.
  def test(self,testList):
    guessList = []
    for testPoint in testList:
      signal = dot(self.weightVec,testPoint[:4])
      if (signal < self.theta):
        guess = 0 #Setosa
        guessList.append(guess)
      else:
        guess = 1 #Not setosa
        guessList.append(guess)
    return guessList

  # Accessor for the weight vector to be used when creating the next perceptron.
  def getWeightVec(self):
    return self.weightVec

  # Validates the guesses of the trained simple feedback perceptron on new data.
  def validate(self, guessList,testList):
    scoreReport = []
    for i in range(len(guessList)):
      if (guessList[i] == 0) and (testList[i][4] == "Iris-setosa"):
        scoreReport.append(1)
      elif (guessList[i] == 1) and (testList[i][4] != "Iris-setosa"):
        scoreReport.append(1)
      else:
        scoreReport.append(0)
      i = i + 1
    return scoreReport

# ------------------------------------------------------------------------------
#                     Algorithm 3:  Error Correction Perceptron

# This perceptron aims to only classify flowers based on membership in all three
# species of flower. It consists of two actual nodes: one to classify the setosa
# species, and one that classifies the versicolor species. Thus it acts as an
# XOR gate. It uses error correction learning, outlined to change the weights when
# a point is missclassified. Because the data is not linearly separable, this
# method was capped at an iteration limit of 10,000. A step function was used as
# the activation function.
# ------------------------------------------------------------------------------

class Ann3_ErrorCorrection(object):
  # Constructor for the error correction perceptron object. All parameters were
  # initialized to 1 for convenience, except the learning rate for the second
  # classifer.
  def __init__(self):
    self.trained = False
    self.weightVec_1 = [1,1,1,1]
    self.learningRate_1 = 1
    self.theta_1 = 1

    self.weightVec_2 = [1,1,1,1,1] # Note that one of the weights is the bias
    self.learningRate_2 = 0.1
    self.theta_2 = 1

  # Trains the error correction perceptron on the given training data. Because
  # the datapoints are not linearly separable, I knew I could not run this
  # algorithm until all points were trained properly--so I set a max iteration
  # cap at 10,000.
  def train(self, trainList):
    runLength2 = 0 #Maximum number of points that have been classified correctly
    index = 0 #Start with the first point
    targetNum = len(trainList)
    guessList = []

    #Train an instance of the first classifier to divide setosa and not setosa
    firstClassifier = Ann2_SimpleFeedback()
    print "Training first classifer..."
    firstClassifier.train(trainList)
    self.weightVec_1 = firstClassifier.getWeightVec()

    print "Training second classifier..."
    iterationNum = 0
    iterationCap = 10000
    while (not self.trained) and (iterationNum<iterationCap):
      #Try with an initial set of weights
      trainingPoint = trainList[index]
      xVec = [trainingPoint[0],trainingPoint[1],trainingPoint[2],trainingPoint[3]]
      xVec2 = [trainingPoint[0],trainingPoint[1],trainingPoint[2],trainingPoint[3],1]
      signal = dot(self.weightVec_1, xVec)

      #Use the trained simple feedback node to make the first classification
      if (signal < self.theta_1):
        guess = 0 #Setosa
        guessList.append(guess)
      else:
        guess = 1 #Not setosa
        guessList.append(guess)

      if (guess == 0) and (trainingPoint[4] == "Iris-setosa"):
        index += 1
        runLength2 += 1
      else:
        #Classify the next level down
        #Make a guess based on the initial weights
        signal = dot(self.weightVec_2, xVec2)
        if (signal < self.theta_2):
          guess = 0 #Versicolor
          guessList.append(guess)
        else:
          guess = 1 #Not Versicolor
          guessList.append(guess)

        #If the guess is correct, keep going
        if (guess == 0) and (trainingPoint[4] == "Iris-versicolor"):
          index += 1
          runLength2 += 1
        elif (guess == 1) and (trainingPoint[4] != "Iris-versicolor"):
          index += 1
          runLength2 += 1
        else: #If guess is incorrect change weights.
          if (trainingPoint[4] == "Iris-versicolor"):
            desired = 0
            self.changeWeights(guess, desired, xVec2)
            iterationNum += 1
          else:
            desired = 1
            self.changeWeights(guess, desired, xVec2)
            iterationNum += 1
          index = 0
          #Do not reset run lengths if it is about to hit cap
          if (iterationNum + 1 != iterationCap - 1):
            runLength2 = 0
      #Check if fully trained
      if (runLength2 == targetNum):
        self.trained = True

    #Determine training accuracy
    if (runLength2 == targetNum):
      print "Done training."
      trainingAccuracy2 = (runLength2/float(targetNum))*100
      print "Training accuracy of second classifier: ",trainingAccuracy2,"%"
    else:
      print "Maximum iteration count reached."
      trainingAccuracy2 = (runLength2/float(targetNum))*100
      print "Training accuracy of second classifier: ",trainingAccuracy2,"%"

  # Using error correction learning, change weights that led to an incorrect
  # output.
  def changeWeights(self, guess, desired, xVec):
    for index in range(len(xVec)):
      change = (desired-guess)*(self.learningRate_2 * xVec[index])
      self.weightVec_2[index] = self.weightVec_2[index] + change

  # Tests the error correction perceptron on new data.
  def test(self,testList):
    guessList = []
    for testPoint in testList:
      signal_1 = dot(self.weightVec_1,testPoint[:4])
      if (signal_1 < self.theta_1):
        guess = 0 #Setosa
        guessList.append(guess)
      else:
        testPoint = testPoint[:4]
        testPoint.append(1)
        signal_2 = dot(self.weightVec_2,testPoint)
        if (signal_2 < self.theta_2):
          guess = 1 #Versicolor
          guessList.append(guess)
        else:
          guess = 2 #Virginica
          guessList.append(guess)
    return guessList

  # Validates the guesses of the trained error correction perceptron on new data.
  def validate(self, guessList,testList):
    scoreReport = []
    for i in range(len(guessList)):
      if (guessList[i] == 0) and (testList[i][4] == "Iris-setosa"):
        scoreReport.append(1)
      elif (guessList[i] == 1) and (testList[i][4] == "Iris-versicolor"):
        scoreReport.append(1)
      elif (guessList[i] == 2) and (testList[i][4] == "Iris-virginica"):
        scoreReport.append(1)
      else:
        scoreReport.append(0)
      i = i + 1
    return scoreReport

# ------------------------------------------------------------------------------
#                                 Main function
# ------------------------------------------------------------------------------
# Runs the program.
def main():
  print "Gathering training and testing data...\n"
  train = []
  test = []
  train = readFile('train.txt')
  test = readFile('test.txt')
  print "Try different algorithms to classify the Iris dataset:\n"
  q1_guessList = q1(test)
  q2_guessList = q2(train, test)
  q3_guessList = q3(train, test)
  writeToFile(test, q1_guessList, q2_guessList,q3_guessList)
  writeSummaryToFile()

main()
