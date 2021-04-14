# Machine Learning with NumPy

## Description
This project implements a variety of fundamental machine learning algorithms in Python using only NumPy. 

After implementing the supervised models, I tested them on simple datasets (datasets obtained from sklearn, tensorflow, etc) using the python framework unittest and expected the algorithms to perform up to par. When available, I compared the performance of my implementations to the standard implementations of an algorithm to ensure correctness. The directory containing the tests for the classifiers is [here](https://github.com/13jk59/Machine-Learning-From-Scratch/tree/master/ML_algorithms/Supervised_Learning/Classifiers/Tests). The directory containing tests for the regressors is [here](https://github.com/13jk59/Machine-Learning-From-Scratch/tree/master/ML_algorithms/Supervised_Learning/Regression/Tests). 

For testing the unsupervised models, I visually examined the output of the models and ensured they were feasible. For k-means clustering, this involved using the classic Iris Setosa dataset and assessing the output clusters that were produced by the model when k=3. For the graphical models, model outputs were produced during training which were verified (ie: as the model continues to train, it should produce better and better outputs). The directory containing the tests for the unsupervised models is [here](https://github.com/13jk59/Machine-Learning-From-Scratch/tree/master/ML_algorithms/Unsupervised_Learning/Tests). 

## Installation 
```
$ git clone https://github.com/13jk59/MachineLearning_Scratch.git
$ cd MachineLearning_Scratch
$ pip3 install . 
```

## Implementations 
### Supervised Learning 
Base Classes (abstract classes which concrete classes implement, keeps code DRY): 
- [k-Nearest Neighbours (kNN)](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Base_Classes/k_nearest_neighbours_base.py)
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
    - [Classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/RandomForestClassifier.py)
    - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/RandomForestRegressor.py)
  - Bagged Forests
    - [Classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/BaggedForestClassifier.py)
    - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Regression/BaggedForestRegression.py)

- [Softmax Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/SoftmaxRegression.py)
- [Gaussian Naive Bayes](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Supervised_Learning/Classifiers/gaussianNaiveBayes.py) 

### Unsupervised Learning:
- [k-Means Clustering](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/k_Means.py)
- [Principal Component Analysis](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/PCA.py)
- [Deep Autoencoder](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/AutoEncoder.py)
- [Restricted Boltzmann Machine](https://github.com/13jk59/MachineLearning_Scratch/blob/master/ML_algorithms/Unsupervised_Learning/restricted_boltzmann_machine.py)
