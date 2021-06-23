from ML_algorithms.Supervised_Learning.Classifiers.SoftmaxRegression import softmax_regression
from ML_algorithms.Neural_Net_Util.optimizer import Adam, RMSProp
from ML_algorithms.Utility.ScoreFunctions import accuracy
from ML_algorithms.Utility.misc import oneHotEncode
from ML_algorithms.Utility.k_Fold_CV import k_fold_CV
from sklearn.datasets import load_iris
from sklearn import preprocessing
import unittest


class SoftmaxTests(unittest.TestCase):
    """ The implementation of the softmax classifier is correct. With
    8 folds, a cross validation score of 94.4% accuracy was achieved.
    With L1 regularization and L2 regularization, the classifier performs
    as expected -> performance is very sensitive to regParameter. If the
    regularization parameter is even slightly high (>0.3), the performance
    for the l1 regularized and l2 regularized softmax regression models
    falter heavily.
    """

    def setUp(self):
        self.x, self.y = load_iris(return_X_y=True)
        self.x = preprocessing.scale(self.x).T
        self.y_encoded = oneHotEncode(self.y)
        self.softmax_model_no_regularization = softmax_regression(
            self.x.shape[0], len(self.y_encoded))

        self.softmax_model_l1_regularization = softmax_regression(
            self.x.shape[0],
            len(self.y_encoded),
            regularization="L1",
            regParameter=0.01)

        self.softmax_model_l2_regularization = softmax_regression(
            self.x.shape[0],
            len(self.y_encoded),
            regularization="L2",
            regParameter=0.01)

        self.k_fold_obj = k_fold_CV()

    def test_softmax_no_reg(self):

        ## Strength of RMSProp shown - get a 6% increase in accuracy w/ it. 99.3% RMSprop and 93.7% normal gradient descent
        kScore_normalGD = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_no_regularization,
            numEpochs=100,
            learn_rate=0.2,
            k=8)

        kScore_RMSProp = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_no_regularization,
            numEpochs=100,
            learn_rate=0.2,
            k=8,
            optim=RMSProp())

        # Adam is the most sensitive out of the three tested and requires the most hyperparameter tuning
        _, train_acc = self.softmax_model_no_regularization.fit(
            self.x,
            self.y_encoded,
            num_epochs=1000,
            learn_rate=0.01,
            optim=Adam(),
            ret_train_loss=True)
        kScore_Adam = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_no_regularization,
            numEpochs=1000,
            learn_rate=0.01,
            k=8,
            optim=Adam())

        self.assertGreaterEqual(train_acc, 0.90)
        self.assertGreaterEqual(kScore_normalGD, 0.90)
        self.assertGreaterEqual(kScore_RMSProp, 0.96)
        self.assertGreaterEqual(kScore_Adam, 0.98)

    def test_softmax_reg(self):

        kScore1 = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_l1_regularization,
            numEpochs=150,
            learn_rate=0.01,
            k=8)

        kScore2 = self.k_fold_obj.getKScore(
            self.x,
            self.y_encoded,
            accuracy,
            self.softmax_model_l2_regularization,
            numEpochs=150,
            learn_rate=0.01,
            k=8)

        self.assertGreaterEqual(kScore1, 0.80, kScore1)
        self.assertGreaterEqual(kScore2, 0.80)


if __name__ == "__main__":
    unittest.main()
