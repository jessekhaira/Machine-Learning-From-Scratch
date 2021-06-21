import numpy as np


class Optimizer(object):
    """ This class is an abstract class that contains methods
    that every optimization algorithm will implement """

    def updateParams(self):
        raise NotImplementedError


class gradientDescent(Optimizer):

    def updateParams(self, params, dparams, learn_rate, epochNum=None):
        for i in range(len(params)):
            params[i] = params[i] - learn_rate * dparams[i]
        return params


class gradientDescentMomentum(Optimizer):

    def __init__(self, beta=0.9):
        self.runningGradients = []
        self.beta = beta

    def updateParams(self, params, dparams, learn_rate, epochNum=None):
        # epoch zero, initialize running gradients for every single parameter in this layer
        if not self.runningGradients:
            for i in range(len(params)):
                self.runningGradients.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            self.runningGradients[i] = self.beta * self.runningGradients[i] + (
                1 - self.beta) * dparams[i]
            params[i] = params[i] - learn_rate * self.runningGradients[i]
        return params


class AdaGrad(Optimizer):

    def __init__(self):
        self.runningGradients = []
        self.eps = 1e-8

    def updateParams(self, params, dparams, learn_rate, epochNum=None):
        if not self.runningGradients:
            for i in range(len(params)):
                self.runningGradients.append(np.zeros_like(dparams[i]))

        for i in range(len(params)):
            # add square of dL/dparam to the running gradient for this param
            self.runningGradients[i] += np.power(dparams[i], 2)
            params[i] = params[i] - (learn_rate * dparams[i]) / (
                np.sqrt(self.runningGradients[i] + self.eps))
        return params


class RMSProp(Optimizer):

    def __init__(self, beta=0.9, eps=1e-8):
        self.runningGradients = []
        self.beta = beta
        self.eps = eps

    def updateParams(self, params, dparams, learn_rate, epochNum=None):
        # epoch zero, initialize running gradients for every single parameter in this layer
        if not self.runningGradients:
            for i in range(len(params)):
                self.runningGradients.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            self.runningGradients[i] = self.beta * self.runningGradients[i] + (
                1 - self.beta) * np.square(dparams[i])
            params[i] = params[i] - (learn_rate * dparams[i]) / (
                np.sqrt(self.runningGradients[i]) + self.eps)

        return params


class Adam(Optimizer):

    def __init__(self, beta1=0.9, beta2=0.9, eps=1e-8):
        self.runningGradients = []
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def updateParams(self, params, dparams, learn_rate, epochNum=None):
        # epoch zero, initialize running gradients for every single parameter in this layer
        if not self.runningGradients:
            for i in range(len(params)):
                self.runningGradients.append(
                    [np.zeros_like(params[i]),
                     np.zeros_like(params[i])])

        for i in range(len(params)):
            momentum = self.runningGradients[i][0]
            adaptiveLR = self.runningGradients[i][1]
            dLdparam = dparams[i]
            # update the running average vectors based on current dL/dparam
            momentum = self.beta1 * momentum + (1 - self.beta1) * dLdparam
            adaptiveLR = self.beta2 * adaptiveLR + (1 - self.beta2) * np.power(
                dLdparam, 2)
            # unbias estimates - important when epochNum is low
            momentum = momentum / (1 - (self.beta1**epochNum))
            adaptiveLR = adaptiveLR / (1 - (self.beta2**epochNum))
            # finally, update params and save new adaptiveLearnRateVec and momentumUpdateVec
            params[i] = params[i] - (learn_rate * momentum) / (
                np.sqrt(adaptiveLR) + self.eps)
            self.runningGradients[i][0] = momentum
            self.runningGradients[i][1] = adaptiveLR

        return params
