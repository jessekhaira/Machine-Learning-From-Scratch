# Machine Learning with NumPy

## Description
This project implements a variety of fundamental machine learning algorithms in Python using only NumPy. 

Every algorithm that I coded out was tested

## Installation 
```
$ git clone https://github.com/13jk59/MachineLearning_Scratch.git
$ cd MachineLearning_Scratch
$ pip3 install . 
```

## Implementations 
### Supervised Learning 
Base Classes (abstract classes which concrete classes implement to keep code DRY): 
- [K-Nearest Neighbours (kNN)](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Base_Classes/kNearestNeighbours_baseClass.py)
- [Neural Network](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Neural_Net_Util/NeuralNetwork_Base.py)
- [Decision Tree](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Base_Classes/DecisionTree.py)
- [Bagged Forest](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Base_Classes/BaggedForest.py)


Deep Learning:
- [Convolutional Neural Network](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/ConvolutionalNeuralNet.py)
- [Recurrent Neural Network](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/RecurrentNet_languageModel.py)
- [MultiLayer Perceptron](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/MultiLayerPerceptron.py)

- Neural Network Layers:
  - [Convolutional Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Neural_Net_Util/ConvolutionalLayers.py) (includes Conv2D, max pooling and average pooling layers) 
  - [Fully Connected Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Neural_Net_Util/NeuralNet_Layers.py) (includes Dense, BatchNorm, Dropout layers)
  - [Recurrent Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Neural_Net_Util/RecurrentNetLayers.py) (includes vanilla RNN cell) 

Models:
- [Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/Linear_Regression.py)
  - Encompasses Polynomial Regression, Ridge Regression and Lasso Regression

- [Logistic Regression & One V All Logistic Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/Logistic_Regression.py)

- kNN
  - [Classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/kNN_classifier.py)
  - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/kNN_regressor.py)

- Decision Trees
  - [Classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/classificationTree.py) 
  - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/RegressionTree.py)

- Tree Ensembles
  - Random Forests
    - [Classifer](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/RandomForestClassifier.py)
    - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/RandomForestRegressor.py)
  - Bagged Forests
    - [Classifer](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/BaggedForestClassifier.py)
    - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/BaggedForestRegression.py)

- [Softmax Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/SoftmaxRegression.py)
- [Gaussian Naive Bayes](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/gaussianNaiveBayes.py) 

### Unsupervised Learning:
- [k-Means Clustering](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/k_Means.py)
- [Principal Component Analysis](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/PCA.py)
- [Deep Autoencoder](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/AutoEncoder.py)
- [Restricted Boltzmann Machine](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/RestrictedBoltzmannMachine.py)
