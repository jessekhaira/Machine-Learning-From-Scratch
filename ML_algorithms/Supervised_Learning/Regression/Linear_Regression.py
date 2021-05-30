import numpy as np
from ML_algorithms.Neural_Net_Util.NeuralNetwork_Base import NeuralNetwork_Base
from ML_algorithms.Neural_Net_Util.LossFunctions import mean_squared_error
from ML_algorithms.Neural_Net_Util.ActivationFunctions import IdentityActivation
from itertools import combinations_with_replacement
from ML_algorithms.Neural_Net_Util.Optimizers import gradientDescent
from sklearn import preprocessing


class Base_Regression(NeuralNetwork_Base):
    """
    This class represents the base class for linear regression, which all linear regression
    classes will inherit. 
    """

    def __init__(self, degree, regularization=None, regParameter=None):
        # Save the degree of the polynomial function that is desired to be fit.
        # Will be used later to transform the input features to the final function features we will fit
        self.degree = degree
        # Loss function for regression tasks is RSS averaged over all examples = MSE
        lossFunction = mean_squared_error(regularization, regParameter)
        super(Base_Regression, self).__init__(lossFunction=lossFunction,
                                              input_features=None)

    def fitGD(self,
              xtrain,
              ytrain,
              xvalid=None,
              yvalid=None,
              num_epochs=10,
              batch_size=32,
              ret_train_loss=False,
              learn_rate=0.01,
              optim=gradientDescent()):
        # the fit method is basically the same as the neural net base, other than the transformation of the features
        # that needs to take place before fitting
        xtrain = self._getPolynomialFeatures(xtrain)
        if xvalid is not None:
            xvalid = self._getPolynomialFeatures(xvalid)
        # Number of features is on the rows, so num_input == len(X_poly)
        self.num_input = len(xtrain)
        # Linear regression models have one layer with one neuron using an identity activation function
        activ = IdentityActivation()
        self.add_layer(1, activ)
        # If ret_train_loss is true, we will return a list of the losses averaged over each epoch for the training set and
        # the validation set

        return self.fit(xtrain,
                        ytrain,
                        xvalid=xvalid,
                        yvalid=yvalid,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        ret_train_loss=ret_train_loss,
                        learn_rate=learn_rate,
                        optim=optim)

    def predictLR(self, X):
        # the predict method is basically the same as the neural net base, other than the transformation of the features
        # to the polynomial features that needs to take place before fitting
        X_poly = self._getPolynomialFeatures(X)
        return self.predict(X_poly)

    def _getPolynomialFeatures(self, dataset):
        # Features on rows, examples on columns
        if self.degree == 1:
            return dataset
        originalNumFeatures = len(dataset)
        # Get combinations of indices of features ex: (0,), (1,), (2,), (0,0), (0,1), (0,2), (1,2), etc
        allCombos = self._getCombos(originalNumFeatures, self.degree)
        numPolynomialFeatures = len(allCombos)
        # Make a empty new data matrix of the appropriate shape
        # We will fill in the rows of this matriix with the appropriate feature values
        new_X = np.empty((numPolynomialFeatures, dataset.shape[1]))
        # Using the combo of the features which is a tuple like (0,0,0)
        # We can say np.prod(dataset[combo_features,:], axis=0) which will do
        # an element by element multiplication along the feature values in
        # each corresponding column
        for feature_idx, combo_features in enumerate(allCombos):
            new_X[feature_idx, :] = np.prod(dataset[combo_features, :], axis=0)
        # Preprocess your new dataset after you've made all the features that you want to use
        new_X = preprocessing.scale(new_X.T)
        return new_X.T

    def _getCombos(self, numFeatures, degree):
        allCombos = [
            combinations_with_replacement(range(numFeatures), i)
            for i in range(1, degree + 1)
        ]
        unrolled = [item for sublist in allCombos for item in sublist]
        return unrolled


class LinearRegression(Base_Regression):

    def __init__(self, degree):
        super(LinearRegression, self).__init__(degree=degree,
                                               regularization=None,
                                               regParameter=None)


class RidgeRegression(Base_Regression):

    def __init__(self, degree, regParam=0.2):
        super(RidgeRegression, self).__init__(degree=degree,
                                              regularization="L2",
                                              regParameter=regParam)


class LassoRegression(Base_Regression):

    def __init__(self, degree, regParam=0.2):
        super(LassoRegression, self).__init__(degree=degree,
                                              regularization="L1",
                                              regParameter=regParam)
