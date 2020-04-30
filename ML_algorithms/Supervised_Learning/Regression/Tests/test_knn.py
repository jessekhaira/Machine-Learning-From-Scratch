import sys
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Supervised_Learning/Regression")
sys.path.append("/Users/jessek/Documents/MachineLearning_Numpy/ML_algorithms/Utility")
from kNN_regressor import k_Nearest_Neighbours
import unittest
import numpy as np 
import sklearn.datasets
from ScoreFunctions import MSE
from ScoreFunctions import MAE
from ScoreFunctions import RMSE 
import sklearn.metrics
import sklearn.model_selection

class tests(unittest.TestCase):
    """
    This class contains unit tests for each of the methods
    defined for the kNN class.
    """
    def testAttributes(self):
        obj1 = k_Nearest_Neighbours()
        self.assertEqual(obj1.k, 10)
        self.assertEqual(obj1.similarity_metric, "L2")
    
    def test_xy_fit(self):
        #generate a np matrix for x_train and y_train and test if it fits properly
        x_train = np.random.randn(10,20)
        y_train = np.random.randn(10,1)
        obj2 = k_Nearest_Neighbours()
        obj2.fit(x_train, y_train)
        self.assertEqual(np.sum(obj2.modelX-x_train), 0)
        self.assertEqual(np.sum(obj2.modelY-y_train), 0)

    def test_overall_model(self):
        # This minimal implementation of kNN
        # should be able to achieve MAE <=10, MSE <= 50 and RMSE <=10 if 
        # everything is wired correctly 
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(X,y)
        ytest = ytest.reshape(-1, 1)
        test_obj = k_Nearest_Neighbours()
        test_obj.fit(xtrain, ytrain)
        prediction_test = test_obj.predict(xtest)
        self.assertEqual(prediction_test.shape, ytest.shape)
        mean_abs = MAE(ytest, prediction_test)
        mean_sq = MSE(ytest, prediction_test)
        root_meansq = RMSE(ytest, prediction_test)
        print(mean_abs)
        print(mean_sq)
        print(root_meansq)
        self.assertLessEqual(mean_abs, 10)
        self.assertLessEqual(mean_sq, 65)
        self.assertLessEqual(root_meansq, 10)


if __name__ == "__main__":
    unittest.main()