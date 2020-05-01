# MachineLearning_Scratch

## Description
This project implements a variety of machine learning algorithms in both classic statistical machine learning
and deep learning in python using only the NumPy framework. 

## Installation 
```
$ git clone https://github.com/13jk59/MachineLearning_Scratch.git
$ cd MachineLearning_Scratch
$ python setup.py install
```

## Implementations 
### Supervised Learning 
Base Classes:
1. [K-NN](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Base_Classes/kNearestNeighbours_baseClass.py)
2. [Neural Net Base](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Neural_Net_Util/NeuralNetwork_Base.py)
3. [Decision Tree](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Base_Classes/DecisionTree.py)
4. [Bagged Forest](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Base_Classes/BaggedForest.py)

Neural Net Layers:
1. [Conv Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Neural_Net_Util/ConvolutionalLayers.py)
2. [Fully Connected Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Neural_Net_Util/NeuralNet_Layers.py)
3. [Recurrent Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Neural_Net_Util/RecurrentNetLayers.py)

Models:
1. [Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/Linear_Regression.py)
- Polynomial Regression
- Ridge Regression
- Lasso Regression
2. [Logistic Regression & One V All Logistic Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/Logistic_Regression.py)

3. kNN
- [k-NN classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/kNN_classifier.py)

- [k-NN regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/kNN_regressor.py)

4. [Classification Tree](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/classificationTree.py) 

5. [Regression Tree](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/RegressionTree.py)

6. Ensemble Trees
- [Random Forest Classifer](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/RandomForestClassifier.py)

- [Random Forest Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/RandomForestRegressor.py)

- [Bagged Forest Classifer](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/BaggedForestClassifier.py)

- [Bagged Forest Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/BaggedForestRegression.py)

7. [Softmax Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/SoftmaxRegression.py)
8. [Convolutional Neural Net](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/ConvolutionalNeuralNet.py)
9. [Recurrent Neural Net](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/RecurrentNet_languageModel.py)
10. [MultiLayer Perceptron](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/MultiLayerPerceptron.py)

### Unsupervised Learning:
1. [k-Means](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/k_Means.py)
2. [Prinicpal Component Analysis](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/PCA.py)
3. [Autoencoder](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/AutoEncoder.py)
