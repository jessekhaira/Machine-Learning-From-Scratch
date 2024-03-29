# Machine Learning with NumPy

## Description

This project implements a variety of fundamental machine learning algorithms in Python with NumPy.

After implementing the supervised models, I tested them on simple datasets (datasets obtained from sklearn, tensorflow, etc) using the python framework unittest and expected the algorithms to perform up to par. When available, I compared the performance of my implementations to the standard implementations of an algorithm to ensure correctness. The directory containing the tests for the classifiers is [here](https://github.com/13jk59/Machine-Learning-From-Scratch/tree/master/machine_learning_algorithms/supervised_learning/classifiers/Tests). The directory containing tests for the regressors is [here](https://github.com/13jk59/Machine-Learning-From-Scratch/tree/master/machine_learning_algorithms/supervised_learning/regression/Tests).

For testing the unsupervised models, I visually examined the output of the models and ensured they were feasible. For k-means clustering, this involved using the classic Iris Setosa dataset and assessing the output clusters that were produced by the model when k=3. For the graphical models, model outputs were produced during training which were verified (ie: as the model continues to train, it should produce better and better outputs). The directory containing the tests for the unsupervised models is [here](https://github.com/13jk59/Machine-Learning-From-Scratch/tree/master/machine_learning_algorithms/unsupervised_learning/Tests).

## Installation

```
$ git clone https://github.com/13jk59/MachineLearning_Scratch.git
$ cd MachineLearning_Scratch
$ pip3 install .
```

## Implementations

### Supervised Learning

Base Classes (abstract classes which concrete classes implement, keeps code DRY):

- [k-Nearest Neighbours (kNN)](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/base_classes/k_nearest_neighbours_base.py)
- [Neural Network](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/neural_net_utility/neural_net_base.py)
- [Decision Tree](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/base_classes/decision_tree.py)
- [Bagged Forest](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/base_classes/bagged_forest.py)

Deep Learning:

- [Convolutional Neural Network](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/convolutional_neural_network.py)
- [Recurrent Neural Network](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/recurrent_network.py)
- [MultiLayer Perceptron](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/multi_layer_perceptron.py)

- Neural Network Layers:
  - [Convolutional Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/neural_net_utility/convolutional_layers.py) (includes Conv2D, max pooling and average pooling layers)
  - [Fully Connected Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/neural_net_utility/neural_net_layers.py) (includes Dense, BatchNorm, Dropout layers)
  - [Recurrent Layers](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/neural_net_utility/reccurent_neural_net_layers.py) (includes vanilla RNN cell)

Models:

- [Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/regression/linear_regression.py)

  - Encompasses Polynomial Regression, Ridge Regression and Lasso Regression

- [Logistic Regression & One V All Logistic Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/logistic_regression.py)

- kNN

  - [Classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/k_nearest_neighbours_classifier.py)
  - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/regression/k_nearest_neighbours_regressor.py)

- Decision Trees

  - [Classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/classification_tree.py)
  - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/regression/regression_tree.py)

- Tree Ensembles

  - Random Forests
    - [Classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/random_forest_classifier.py)
    - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/regression/random_forest_regressor.py)
  - Bagged Forests
    - [Classifier](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/bagged_forest_classifier.py)
    - [Regressor](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/regression/bagged_forest_regressor.py)

- [Softmax Regression](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/softmax_regression.py)
- [Gaussian Naive Bayes](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/supervised_learning/classifiers/gaussian_naive_bayes.py)

### Unsupervised Learning:

- [K-Means](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/unsupervised_learning/k_means.py)
- [Principal Component Analysis](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/unsupervised_learning/principal_component_analysis.py)
- [Deep Autoencoder](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/unsupervised_learning/auto_encoder.py)
- [Restricted Boltzmann Machine](https://github.com/13jk59/MachineLearning_Scratch/blob/master/machine_learning_algorithms/unsupervised_learning/restricted_boltzmann_machine.py)
