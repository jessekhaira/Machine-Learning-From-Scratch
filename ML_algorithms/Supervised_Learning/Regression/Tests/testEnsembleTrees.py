import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Regression")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
from BaggedForestRegression import BaggedForestRegression
from RandomForestRegressor import RandomForestRegressor
import unittest
import numpy as np 
import sklearn
from sklearn.datasets import load_boston


X1, Y1 = load_boston(return_X_y=True)
X1 = X1.T
Y1 = Y1.T.reshape(1, -1)

class tests(unittest.TestCase):
    def testRF_Regressor(self):
        # Use 5 random bootstrapped samples to train each tree and then get OOB mse and rmse
        # should be quite low 
        mod2= RandomForestRegressor(verbose= True, bootstrap= True, max_samples=5, minSamplesSplit=1, maxFeatures=int(X1.shape[0]**0.5))
        mod2.fit(X1, Y1)
        mse2, rmse2 = mod2.getOOBScore(X1,Y1)
        print(mse2, rmse2)
        self.assertGreaterEqual(mse2, 20)
        self.assertLessEqual(rmse2, 10)
        

        # fit on a slightly bigger portion and see what happens - should be better than before
        mod4= RandomForestRegressor(verbose= True, bootstrap= True, max_samples=15, minSamplesSplit=3, maxFeatures=int(X1.shape[0]**0.5))
        mod4.fit(X1,Y1)
        mse4, rmse4 = mod4.getOOBScore(X1,Y1)
        print(mse4, rmse4)
        self.assertLessEqual(mse4, mse2)
        self.assertLessEqual(rmse4, rmse2)

        # allow each tree to see a large bootstrapped sample and get mse and rmse - should perform quite well 
        # may not perform as well as just one decision tree on this particular dataset since the training procedure has so much stochasiticy in it 
        # and it is already difficult to overfit to this dataset 
        mod5= RandomForestRegressor(verbose= True, bootstrap= True, max_samples=300, minSamplesSplit=25, maxFeatures=int(X1.shape[0]**0.5), maxDepth=4, min_impurity_decrease=0.25)
        mod5.fit(X1,Y1)
        mse5, rmse5 = mod5.getOOBScore(X1,Y1)
        print(mse5, rmse5)
        self.assertLessEqual(mse5, mse4)
        self.assertLessEqual(rmse5, rmse4)


    def testBaggedForestRegressor(self):
        ## Diff b/w bagged forest and random forest - bagged forests sees every single feature at every single split point
        # can achieve lower bias since there is naturally less stochasiticy in the training procedure, but will struggle with overfitting
        # and lack of variety in the ensemble 
        mod2= BaggedForestRegression( verbose= True, bootstrap= True, max_samples=5, minSamplesSplit=1)
        mod2.fit(X1, Y1)
        mse2, rmse2 = mod2.getOOBScore(X1,Y1)
        print(mse2, rmse2)
        self.assertGreaterEqual(mse2, 0.90)
        self.assertGreaterEqual(rmse2, 0.05)
        

        # fit on a slightly bigger portion and see what happens - should be better than before
        mod4= BaggedForestRegression(verbose= True, bootstrap= True, max_samples=15, minSamplesSplit=3)
        mod4.fit(X1,Y1)
        mse4, rmse4 = mod4.getOOBScore(X1,Y1)
        print(mse4, rmse4)
        self.assertLessEqual(mse4, mse2)
        self.assertLessEqual(rmse4, rmse2)

        # allow each tree to see a large bootstrapped sample and get mse and rmse - should performed quite well 
        # may not perform as well as just one decision tree on this particular dataset since the training procedure has so much stochasicity in it 
        # and it is already difficult to overfit to this dataset 
        mod5= BaggedForestRegression( verbose= True, bootstrap= True, max_samples=150, minSamplesSplit=20)
        mod5.fit(X1,Y1)
        mse5, rmse5 = mod5.getOOBScore(X1,Y1)
        print(mse5, rmse5)
        self.assertLessEqual(mse5, mse4)
        self.assertLessEqual(rmse5, rmse4)




if __name__ == "__main__":
    unittest.main()