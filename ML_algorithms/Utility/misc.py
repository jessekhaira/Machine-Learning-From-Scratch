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

