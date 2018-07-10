# Metamorphic testing
# Relations and their explanations may be found at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3019603/

import p3b1_baseline_keras2

import keras
import numpy as np
import copy

import unittest

class p3b1tests(unittest.TestCase):

    # Note: test cases only run when they start with 'test'
    (features_train, truths_train, features_test, truths_test) = p3b1_baseline_keras2.readData()
    _srcModel = p3b1_baseline_keras2.do_10_fold(features_train, truths_train, features_test, truths_test,DeterministicResults=True)
    _origPredictions = _srcModel[0][1]
        #@classmethod
        #def setUpClass(self):
        
        #truth0.extend( ret[ 0 ][ 0 ] )
        #pred0.extend( ret[ 0 ][ 1 ] )
        #truth1.extend( ret[ 1 ][ 0 ] )
        #pred1.extend( ret[ 1 ][ 1 ] )
        #truth2.extend( ret[ 2 ][ 0 ] )
        #pred2.extend( ret[ 2 ][ 1 ] )

    def setUp(self):
        print('\nStarting next test...\n')

    def test_DeterministicResults(self):
        newModel = p3b1_baseline_keras2.do_10_fold(self.features_train, self.truths_train, self.features_test, self.truths_test,DeterministicResults=True)
        assert np.array_equal(newModel[0][1],self._srcModel[0][1])

    def test_MR0_ConsistenceWithAffineTransform(self):
        print('MR 0')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        numFeatures = X_train[0].shape[1]
        
        #Makes an array of random size filled with random numbers between 0 and numFeatures without repitition
        randomSubset = np.random.choice(range(numFeatures),np.random.randint(numFeatures),False)

        k = np.random.randint(1,100)
        b = np.random.randint(100)
        print(len(X_train))
        for x in range(len(X_train)):
            for i in range(randomSubset.shape[0]):
                c = randomSubset[i]
                X_train[x][:,c] = X_train[x][:,c]*k + b
                X_test[x][:,c] = X_test[x][:,c]*k + b

        transformedPredictions = p3b1_baseline_keras2.do_10_fold(X_train,y_train,X_test,y_test,True)
        print(transformedPredictions[0][1])
        print(self._srcModel[0][1])
        assert np.array_equal(self._origPredictions,transformedPredictions[0][1]), "affine transformations change the outcome of the model"


    def test_MR11_PermutationOfClassLabels(self):
        print('MR 1.1')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()

        print(y_test[0])
        (y_train[0], y_test[0], p) = self.__permuteLabels(y_train[0],y_test[0])
        print(p)
        print(y_test[0])

        newModel = p3b1_baseline_keras2.do_10_fold(X_train,y_train,X_test,y_test,True)
        newPreds = newModel[0][1]
        print(self._origPredictions)
        print(p)
        print(newPreds)
        for x in range(X_test[0].shape[0]):
            print(newPreds[x])
            print(p[self._origPredictions[x]])
            assert newPreds[x] == p[self._origPredictions[x]], "permuting class labels changes outcome of model"
        
    def test_MR12_PermutationOfAttributes(self):
        print('MR 1.2')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        X = [X_train[0], X_train[1], X_train[2], X_test[0], X_test[1], X_test[2]]
        
        X = self.__shuffleColumnsInUnison(X)
        
        for x in range(len(X_train)):
            print(X_train[x].shape)

        X_train[0] = X[0]
        X_train[1] = X[1]
        X_train[2] = X[2]
        X_test[0] = X[3]
        X_test[1] = X[4]
        X_test[2] = X[5]

        shuffledModelPredictions = (p3b1_baseline_keras2.do_10_fold(X_train,y_train,X_test,y_test,True))[0][1]

        assert np.array_equal(self._origPredictions,shuffledModelPredictions), "permuting the order of the features changes the outcome of the model"
    
    def test_MR21_AddUninformativeAttribute(self):
        print('MR 2.1')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        
        for i in range(len(X_train)):
            X_train[i] = np.column_stack( [ X_train[i] , np.zeros(X_train[i].shape[0]) ] )
        for i in range(len(X_test)):
            X_test[i] = np.column_stack( [ X_test[i] , np.zeros(X_test[i].shape[0]) ] )


        newPreds = (p3b1_baseline_keras2.do_10_fold(X_train,y_train,X_test,y_test,True))[0][1]
        assert np.array_equal(self._origPredictions,newPreds), "adding an uninformative attribute changes outcome"
   
    def test_MR22_AddInformativeAttribute(self):
        print('MR 2.2')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        
        #pick a random class
        n = np.random.randint(len(y_test[0]))

        for i in range(len(X_train)):
            X_train[i] = np.column_stack( [ X_train[i] , np.zeros(X_train[i].shape[0]) ] )
        for i in range(len(X_test)):
            X_test[i] = np.column_stack( [ X_test[i] , np.zeros(X_test[i].shape[0]) ] )

        for i in range(len(X_train)):
            for j in range(len(y_train[i])):
                if y_train[i][j] == n:
                    X_train[i][j,-1] = 1
        
        for i in range(len(X_test)):
            for j in range(len(y_test[i])):
                if y_test[i][j] == n:
                    X_test[i][j,-1] = 1 

        newPreds =  (p3b1_baseline_keras2.do_10_fold(X_train,y_train,X_test,y_test,True))[0][1]
        
        for x in range(len(self._origPredictions)):
            if (self._origPredictions[x] == n):
                assert newPreds[x] == n, "adding an informative feature for class n changed previous classifications of n to another class"
   
    def test_MR31_ConsistenceWithRePrediction(self):
        print('MR 3.1')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        n = np.random.randint(len(y_test[0]))
        nPred = self._origPredictions[n]

        X_train[0] = np.vstack([X_train[0],X_test[0][n]]) 
        y_train[0] = np.append(y_train[0], y_test[0][n])

        newPreds =  (p3b1_baseline_keras2.do_10_fold(X_train,y_train,X_test,y_test,True))[0][1]
        
        assert newPreds[n] == nPred

    def test_MR32_AdditionalTrainingSample(self):
        print('MR 3.2')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        n = np.random.randint(max(y_test[0])+1)

        newX_train = copy.copy(X_train)
        newY_train = copy.copy(y_train)

        for i in range(y_train[0].shape[0]):
            if y_train[0][i] == n:
                newX_train[0] = np.vstack([newX_train[0],X_train[0][i]])
                newY_train[0] = np.append(newY_train[0],y_train[0][i])

        newPreds = (p3b1_baseline_keras2.do_10_fold(newX_train,newY_train,X_test,y_test,True))[0][1]

        for x in range(len(self._origPredictions)):
            if (self._origPredictions[x] == n):
                assert newPreds[x] == n, "doubling the training samples for class n changed some classifications from n to another class"
   
    def test_MR41_AddClassByDuplicatingSamples(self):
        print('MR 4.1')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        numClasses = max(y_test[0])+1
        cl = np.random.randint(numClasses)
        newX_train = copy.copy(X_train)
        newY_train = copy.copy(y_train)
        for x in range(len(y_train[0])):
            if y_train[0][x] != cl:
                newX_train[0] = np.vstack([newX_train[0],X_train[0][x]])
                newY_train[0] = np.append(newY_train[0],y_train[0][x])
                newY_train[0][-1] = newY_train[0][x] + numClasses #give it the new label

        print(y_train[0].shape)
        print(y_train[0])
        print(newY_train[0].shape)
        print(newY_train[0])

        newPreds = (p3b1_baseline_keras2.do_10_fold(newX_train,newY_train,X_test,y_test,True))[0][1]

        for x in range(len(self._origPredictions)):
            if (self._origPredictions[x] == cl):
                assert newPreds[x] == cl, "adding new classes by doubling the training samples for classes other than n made our classifier worse for class n"

    def test_MR42_AddClassesByReLabelingSamples(self):
        print('MR 4.2')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        numClasses = max(y_test[0])+1
        cl = np.random.randint(numClasses)

        newYTrain = copy.copy(y_train)

        for x in range(len(y_train[0])):
            if y_train[0][x] != cl:
                if np.random.random() < .5:
                   y_train[0][x] = y_train[0][x] + numClasses
            
        newPreds = (p3b1_baseline_keras2.do_10_fold(X_train,newYTrain,X_test,y_test,True))[0][1]

        for x in range(len(self._origPredictions)):
            if (self._origPredictions[x] == cl):
                assert newPreds[x] == cl, "relabeling the class of training samples for classes besides n changed some classifications of n"

    def test_MR51_RemovalOfClasses(self):
        print('MR 5.1')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        numClasses = max(y_test[0])+1
        #Class to remove
        cl = np.random.randint(numClasses)
        newXTrain = []
        newYTrain = []

        for x in range(X_train[0].shape[0]):
            if y_train[0][x] != cl:
                newXTrain.append(X_train[0][x])
                newYTrain.append(y_train[0][x])

        X_train[0] = np.asarray(newXTrain)
        y_train[0] = np.asarray(newYTrain)

        newPreds = (p3b1_baseline_keras2.do_10_fold(X_train,y_train,X_test,y_test,True))[0][1]

        for x in range(len(self._origPredictions)):
            if self._origPredictions[x] != cl:
                assert self._origPredictions[x] == newPreds[x], "removing a class causes the model to change predictions for the remaining classes"


    def test_MR52_RemovalOfSamples(self):
        print('MR 5.2')
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        numClasses = max(y_test[0])+1
        #Class to keep
        cl = np.random.randint(numClasses)
        
        samplesToKeep = np.zeros(X_train[0].shape[0])

        for x in range(X_train[0].shape[0]):
            if (np.random.random() < .5) or y_train[0][x] == cl:
                samplesToKeep[x] = 1

        newXTrain = []
        newYTrain = []

        for x in range(X_train[0].shape[0]):
            if samplesToKeep[x]:
                newXTrain.append(X_train[0][x])
                newYTrain.append(y_train[0][x])

        X_train[0] = np.asarray(newXTrain)
        y_train[0] = np.asarray(newYTrain)

        newPreds = (p3b1_baseline_keras2.do_10_fold(X_train,y_train,X_test,y_test,True))[0][1]
        print(self._origPredictions)
        print(newPreds)

        for x in range(len(self._origPredictions)):
            if self._origPredictions[x] == cl:
                assert self._origPredictions[x] == newPreds[int(np.sum(samplesToKeep[:x]))], "Removing samples from classes other than n causes classifications of n to change to other classes"


    def __shuffleColumnsInUnison(self, a):
        p = (np.random.permutation(a[0].shape[1])).astype(int)

        for x in range(len(a)):
            a[x] = np.transpose(a[x])
            assert a[0].shape[0] == a[x].shape[0]
        
        for x in range(len(a)):
            a[x] = np.transpose(a[x][p])
        
        return a

    def __permuteLabels(self,y_train,y_test):
        p = np.arange(np.max(y_test)+1)
        np.random.shuffle(p)
        y_train = p[y_train.astype(int)]
        y_test = p[y_test.astype(int)]
        return (y_train, y_test, p)


    def __getCopiesOfData(self):
        X_train =[copy.copy(self.features_train[0]),copy.copy(self.features_train[1]),copy.copy(self.features_train[2])] 
        X_test = [copy.copy(self.features_test[0]),copy.copy(self.features_test[1]),copy.copy(self.features_test[2])]
        y_train = [copy.copy(self.truths_train[0]),copy.copy(self.truths_train[1]),copy.copy(self.truths_train[2])]
        y_test = [copy.copy(self.truths_test[0]),copy.copy(self.truths_test[1]),copy.copy(self.truths_test[2])]
        return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    unittest.main()