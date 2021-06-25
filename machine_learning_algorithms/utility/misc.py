import numpy as np 

def oneHotEncode(y):
    # Squish y to be one row 
    y_use = y.reshape(-1)
    num_labels = len(np.unique(y_use))
    num_examples = len(y_use)
    output_matrix = np.zeros((num_examples, num_labels))
    output_matrix[np.arange(num_examples), y_use] = 1 
    return output_matrix.T


def oneHotEncodeFeature(numFeatures, idxOne):
    vector = np.zeros((numFeatures,1))
    vector[idxOne] = 1 
    return vector 

def convertToHighestPred(arr):
    arr = np.argmax(arr, axis=0)
    return arr

def euclideanDistance(x, y):
    # euclidean distance is the l2 norm of the vector x- y 
    return np.linalg.norm(x-y, ord=2)

def findRowColMaxElem(tensor):
    idxs = np.unravel_index(np.nanargmax(tensor), tensor.shape)
    return idxs 


def gradientClipping(dparams):
    for gradient_tensor in dparams:
        np.clip(gradient_tensor, -5,5, out = gradient_tensor)

def getUniqueChars(txtFile):
    return list(set(txtFile))

def mapidxToChar(chars):
    return {i:char for i,char in enumerate(chars)}

def mapcharToIdx(chars):
    return {char:idx for idx,char in enumerate(chars)}

def getCovarianceMatrix(matrix):
    mean_features = np.mean(matrix, axis=1, keepdims=True)
    # vectorize operation to get covariance matrix - don't want to do an expensive python for loop
    num_examples = matrix.shape[1]
    return 1/(num_examples-1) * (matrix-mean_features).dot((matrix-mean_features).T)



