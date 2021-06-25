from machine_learning_algorithms.supervised_learning.classifiers.Logistic_Regression import LogisticRegression
import unittest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LR
from machine_learning_algorithms.Utility.ScoreFunctions import accuracy
"""
A few observation from testing the code:
- Increasing regularization parameter leaves normal logistic regression losses untouched
as expected. Avoiding silly bugs :D! 

- Increasing regularization parameter really high, say 5-10 - 100 - 1000, causes the
loss for the L2 regularized and L1 regularized models to be really high. This should happen
since the loss = data_loss + (reg_param) * (L1/Squared L2 norm of weight vector) and if reg
param is high, loss should be high as well.

- Increasing the learning rate while having a high-ish regularization parameter can cause 
divergence pretty quickly.

- If the regularization parameters are kept low, the solutions for the weight vectors will 
be similar between the regularized and unregularized models, and the losses will be similar as well.
This make sense since as if the reg parameter is low, then that means that the majority of the loss
will come from the data and not from the regularization term, likewise with the gradients. 

The norms of the learned vectors should STILL be lower though for the 
L1 regularized and L2 regularized models, which is confirmed. 

- Compared to L2 regularization, L1 regularization produces a much sparser solution 
as can be seen by looking at the norm of the corresponding learned weight vector. This theoretically
makes sense. 

Visually inspecting the weights also sanity checks that: Weight vector learned for unregularized 
model is denser than the model for L2 regularization, which is denser than the weights learned
for L1 regularization. 

With the final test, comparing to sklearns implementation of regularized logistic regression,
the model performed exactly the same when given equivalent values for C (=1/reg_param) and reg_param.
"""
##-- MANUAL TEST W/ Step through debugging----
X, Y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.33,
                                                    random_state=42)

# Standardize train set and test set differently - at inference time, you will not
# be normalizing your input with the output
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test).T
y_test = y_test.T.reshape(1, -1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                      y_train,
                                                      test_size=0.15,
                                                      random_state=42)

X_train = X_train.T
X_valid = X_valid.T
y_train = y_train.T.reshape(1, -1)
y_valid = y_valid.T.reshape(1, -1)

obj1 = LogisticRegression(X_train.shape[0],
                          classificationThreshold=0.5,
                          regularization="L2",
                          regParameter=50)
obj2 = LogisticRegression(X_train.shape[0], classificationThreshold=0.5)
obj3 = LogisticRegression(X_train.shape[0],
                          classificationThreshold=0.5,
                          regularization="L1",
                          regParameter=50)

train_loss1, valid_loss1, train_acc1, valid_acc1 = obj1.fit(X_train,
                                                            y_train,
                                                            X_valid,
                                                            y_valid,
                                                            ret_train_loss=True,
                                                            num_epochs=100,
                                                            learn_rate=0.1)
train_loss2, valid_loss2, train_acc2, valid_acc2 = obj2.fit(X_train,
                                                            y_train,
                                                            X_valid,
                                                            y_valid,
                                                            ret_train_loss=True,
                                                            num_epochs=100,
                                                            learn_rate=0.1)
train_loss3, valid_loss3, train_acc3, valid_acc3 = obj3.fit(X_train,
                                                            y_train,
                                                            X_valid,
                                                            y_valid,
                                                            ret_train_loss=True,
                                                            num_epochs=100,
                                                            learn_rate=0.1)

print('\n')
print(train_loss1)
print('\n')
print(valid_loss1)
print('\n')
print(np.linalg.norm(obj1.layers[0].W)**2)
print(obj1.layers[0].W)

print(train_loss2)
print('\n')
print(valid_loss2)
print('\n')
print(obj2.layers[0].W)
print(np.linalg.norm(obj2.layers[0].W)**2)

print('\n')
print(train_loss3)
print('\n')
print(valid_loss3)
print('\n')
print(obj3.layers[0].W)
print(np.linalg.norm(obj3.layers[0].W, ord=1))

## Compare to Sklearn L2 regularized logistic regression
sklearn_mode = LR(C=1)
sklearn_mode.fit(X_train.T, y_train.ravel())
preds_sk = sklearn_mode.predict(X_test.T)
print(accuracy(y_test, preds_sk))

obj4 = LogisticRegression(X_train.shape[0],
                          classificationThreshold=0.5,
                          regularization="L2",
                          regParameter=1)
obj4.fit(X_train, y_train, num_epochs=50, learn_rate=0.1)
preds_model = obj4.predict(X_test)
print(preds_model)
print(accuracy(y_test, preds_sk))

## 99.4% acccuracy for obj4! :D Which is approx 3.2% higher than the unregularized
# version!!

## Just further verification -- should be similar accuracies
sklearn_mode = LR(C=0.0021)
sklearn_mode.fit(X_train.T, y_train.ravel())
preds_sk = sklearn_mode.predict(X_test.T)
print(accuracy(y_test, preds_sk))

obj5 = LogisticRegression(X_train.shape[0],
                          classificationThreshold=0.5,
                          regularization="L2",
                          regParameter=476)
obj5.fit(X_train, y_train, num_epochs=50, learn_rate=0.1)
preds_model = obj5.predict(X_test)
print(accuracy(y_test, preds_sk))
