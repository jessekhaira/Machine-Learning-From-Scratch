""" This module contains unit tests for the linear regression
algorithm """
import unittest
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
from machine_learning_algorithms.supervised_learning.regression.linear_regression import LinearRegression
from machine_learning_algorithms.supervised_learning.regression.linear_regression import LassoRegression
from machine_learning_algorithms.supervised_learning.regression.linear_regression import RidgeRegression
from machine_learning_algorithms.utility.ScoreFunctions import RMSE, R_squared
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class TestLinearRegression(unittest.TestCase):

    def setUp(self) -> None:
        x, y = sklearn.datasets.load_boston(return_X_y=True)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.10, random_state=42)

        self.x_train = preprocessing.scale(self.x_train)
        self.x_test = preprocessing.scale(self.x_test).T
        self.y_test = self.y_test.T.reshape(1, -1)
        self.x_train, self.x_valid, self.y_train, self.y_valid = (
            train_test_split(self.x_train,
                             self.y_train,
                             test_size=0.10,
                             random_state=42))
        self.x_train = self.x_train.T
        self.x_valid = self.x_valid.T
        self.y_train = self.y_train.T.reshape(1, -1)
        self.y_valid = self.y_valid.T.reshape(1, -1)
        return super().setUp()

    def test1(self) -> None:
        lr_obj = LinearRegression(degree=1)
        lr_obj.fit_iterative_optimizer(xtrain=self.x_train,
                                       ytrain=self.y_train,
                                       xvalid=self.x_valid,
                                       yvalid=self.y_valid,
                                       num_epochs=100,
                                       ret_train_loss=True,
                                       learn_rate=0.1)
        preds = lr_obj.predict_linear_regression(self.x_test)
        r_squared_val = R_squared(self.y_test, preds)
        self.assertGreaterEqual(r_squared_val, 0.5)

    def test2(self):
        lasso_obj = LassoRegression(degree=1, regParam=1000)
        lasso_obj.fit_iterative_optimizer(xtrain=self.x_train,
                                          ytrain=self.y_train,
                                          xvalid=self.x_valid,
                                          yvalid=self.y_valid,
                                          num_epochs=100,
                                          ret_train_loss=True,
                                          learn_rate=0.1)
        preds = lasso_obj.predict_linear_regression(self.x_test)
        r_squared_val = R_squared(self.y_test, preds)
        self.assertGreaterEqual(r_squared_val, 0.5)

    def test3(self):
        ridge_obj = RidgeRegression(degree=1, regParam=1000)
        train_loss3, valid_loss3, train_acc3, valid_acc3 = ridge_obj.fit_iterative_optimizer(
            xtrain=self.x_train,
            ytrain=self.y_train,
            xvalid=self.x_valid,
            yvalid=self.y_valid,
            num_epochs=100,
            ret_train_loss=True,
            learn_rate=0.1)
        print(train_loss3, valid_loss3)
        print(ridge_obj.layers[0].W.T)
        preds2 = ridge_obj.predict_linear_regression(self.x_test)
        print(R_squared(self.y_test, preds2))
        print(np.linalg.norm(ridge_obj.layers[0].W, ord=1))
        print(np.linalg.norm(ridge_obj.layers[0].W, ord=2)**2)
        print('\n')

        ## Linear Reg ##

        ##SKLEARN LASSO##
        # Training for 100 epochs w/ a learning rate of 0.15 gets the exact same R2 score between the models.
        # With a bit hyperparameter tuning, only training for 50 epochs with a learning rate of 0.15, the
        # R2 score performance increases and my model outperforms the sklearn model slightly (0.692 to 0.687).
        lin_reg = sklearn.linear_model.LinearRegression()
        lin_reg.fit(self.x_train.T, self.y_train.ravel())
        preds_linreg = lin_reg.predict(self.x_test.T)
        print(R_squared(self.y_test, preds_linreg))

        lin_regOwn = LinearRegression(degree=1)
        lin_regOwn.fit_iterative_optimizer(xtrain=self.x_train,
                                           ytrain=self.y_train,
                                           num_epochs=50,
                                           learn_rate=0.15)
        preds_lrOwn = lin_regOwn.predict_linear_regression(self.x_test)
        print(R_squared(self.y_test, preds_lrOwn))

        print('\n')

        # These models are estimated differently and hence can't be compared exactly
        # but with a reg paramter of 1 and training for 15 epochs at a leaerning rate of 15,
        # my implementation out performs the implementation from sklearn by a significant margin: 0.703 sklearn, 0.757 own
        lasso_sk = sklearn.linear_model.Lasso(alpha=1)
        lasso_sk.fit(self.x_train.T, self.y_train.ravel())
        preds_lassosk = lasso_sk.predict(self.x_test.T)
        print(R_squared(self.y_test, preds_lassosk))

        lasso_obj2 = LassoRegression(degree=1, regParam=1)
        lasso_obj2.fit_iterative_optimizer(xtrain=self.x_train,
                                           ytrain=self.y_train,
                                           num_epochs=15,
                                           learn_rate=0.15)
        preds_lasso = lasso_obj2.predict_linear_regression(self.x_test)
        print(R_squared(self.y_test, preds_lasso))

        print('\n')
        ##SKLEARN RIDGE##

        # setting alpha = 1000 and training for 200 epochs with a learning
        # rate of 0.1 gets the exact same R2 score for both models of 0.624. Pretty cool!
        ridge_sk = sklearn.linear_model.Ridge(alpha=1000)
        ridge_sk.fit(self.x_train.T, self.y_train.ravel())
        preds_ridgesk = ridge_sk.predict(self.x_test.T)
        print(R_squared(self.y_test, preds_ridgesk))

        ridge_obj2 = RidgeRegression(degree=1, regParam=1000)
        ridge_obj2.fit_iterative_optimizer(xtrain=self.x_train,
                                           ytrain=self.y_train,
                                           num_epochs=200,
                                           learn_rate=0.1)
        preds_ridge = ridge_obj2.predict_linear_regression(self.x_test)
        print(R_squared(self.y_test, preds_ridge))

        ## Polynomial Regression ##
        # Exact same as other models, except instead of fitting a linear function, we can fit polynomial
        # functions with an abritrary degree

        # You have to be super careful with the learning rate here or else you will diverge.
        print('\n')
        degree_2 = LinearRegression(degree=2)
        train_loss = degree_2.fit_iterative_optimizer(xtrain=self.x_train,
                                                      ytrain=self.y_train,
                                                      num_epochs=275,
                                                      learn_rate=0.01,
                                                      ret_train_loss=True)
        print(train_loss)
        deg_2 = degree_2.predict_linear_regression(self.x_test)
        print(R_squared(self.y_test, deg_2))

        print(RMSE(self.y_test, deg_2))

        print('\n')
        lasso_objd2 = LassoRegression(degree=2, regParam=55)
        lasso_objd2.fit_iterative_optimizer(xtrain=self.x_train,
                                            ytrain=self.y_train,
                                            num_epochs=275,
                                            learn_rate=0.01)
        preds_lassod2 = lasso_objd2.predict_linear_regression(self.x_test)
        print(R_squared(self.y_test, preds_lassod2))
        print(RMSE(self.y_test, preds_lassod2))

        print('\n')
        ridge_objd2 = RidgeRegression(degree=2, regParam=55)
        ridge_objd2.fit_iterative_optimizer(xtrain=self.x_train,
                                            ytrain=self.y_train,
                                            num_epochs=275,
                                            learn_rate=0.01)
        preds_ridged2 = ridge_objd2.predict_linear_regression(self.x_test)
        print(R_squared(self.y_test, preds_ridged2))
        print(RMSE(self.y_test, preds_ridged2))


if __name__ == "__main__":
    unittest.main()
