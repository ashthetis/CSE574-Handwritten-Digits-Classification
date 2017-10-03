'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
from scipy.optimize import minimize
from math import sqrt
import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))  # each row represnts weight matrix for one hidden node
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0.0

    # Your code here
    Objective = 0.0
    grad_w1 = 0.0
    grad_w2  = 0.0
    trainingDataSize = training_data.shape[0]
    training_data = np.append(training_data,np.ones([len(training_data),1]),1)  #add column
    training_data=training_data.T

    hiddenLayerOutput = np.dot(w1,training_data)  
    hiddenLayerOutput = sigmoid(hiddenLayerOutput)
    
    hiddenOutputIncludingBiasTerm=hiddenLayerOutput.T
    hiddenOutputIncludingBiasTerm = np.append(hiddenOutputIncludingBiasTerm,np.ones([hiddenOutputIncludingBiasTerm.shape[0],1]),1)  #add column  
    output = np.dot(w2,hiddenOutputIncludingBiasTerm.T)
    output=sigmoid(output) #k*1
    outputclass = np.zeros((n_class,training_data.shape[1])) #initialize all output class to 0

    i=0
    for i in range(len(training_label)):
        label=0
        label= int(training_label[i])
        outputclass[label,i] = 1 # set class of true label
    

     #negative log-likelihood error
    Objective += np.sum(outputclass * np.log(output) + (1.0-outputclass) * np.log(1.0-output))
    deltaOutput = output - outputclass  #k*1
    

    #grad_w2 = grad_w2 + (deltaOutput.reshape((n_class,1)) * np.hstack((hiddenLayerOutput,np.ones(1))))
    grad_w2 =  np.dot(deltaOutput.reshape((n_class,training_data.shape[1])), hiddenOutputIncludingBiasTerm)
    outputDeltaSum = np.dot(deltaOutput.T,w2)


    outputDeltaSum = outputDeltaSum[0:outputDeltaSum.shape[0], 0:outputDeltaSum.shape[1]-1]
    delta_hidden = ((1.0-hiddenLayerOutput) * hiddenLayerOutput*outputDeltaSum.T)
    grad_w1 =  np.dot(delta_hidden.reshape((n_hidden,training_data.shape[1])) , (training_data.T))


    Objective = ((-1)*Objective)/trainingDataSize
    randomization = np.sum(np.sum(w1**2)) + np.sum(np.sum(w2**2))
    Objective = Objective + ((lambdaval * randomization) / (2.0*trainingDataSize))
    grad_w1 = (grad_w1 + lambdaval * w1) / trainingDataSize      #equation 16
    grad_w2 = (grad_w2 + lambdaval * w2) / trainingDataSize      #equation 17    
    obj_val = Objective
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #print(obj_val)
    return (obj_val, obj_grad)


# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = []
    # Your code here
    for testingData in data:
        inputIncludingBiasTerm=np.hstack((testingData,np.ones(1.0)))  #size (d+1)*1
        hiddenLayerOutput = np.dot(w1,inputIncludingBiasTerm)
        hiddenLayerOutput = sigmoid(hiddenLayerOutput)
        
        
        hiddenOutputIncludingBiasTerm=np.hstack((hiddenLayerOutput,np.ones(1.0)))  #size (d+1)*1
        output = np.dot(w2,hiddenOutputIncludingBiasTerm)
        output=sigmoid(output) #k*1
        labels.append(np.argmax(output,axis=0))
        
    labels = np.array(labels)
    # Return a vector with labels   
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y, features

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label, features = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :100}    # Preferred value.

       
now_time = time.time()
start_time = time.strftime("%X")
print(start_time)

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

end_time = time.strftime("%X")
print(end_time)
execution_time = str(round(time.time() - now_time, 2))
print(execution_time)