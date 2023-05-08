"""
Simple Neural Network Implementation
Spencer R. Karofsky
"""

# Libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def sigmoid(x): # 1/1+e^-x
    return 1/ (1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def tanh(x):
    y = np.tanh(x)
    return y

# Loss Functions
'''
Mean Squared Error:
 * Average of the sum of the squares of the differences (residuals) between expected and predicted values
 * mean squared error = (SUM(predicted-target)^2)/n
 * Implementation sourced from GeeksForGeeks.com
'''
def mse(targets,preds): # mean squared error
    mse = np.square(np.subtract(preds,targets)).mean()
    return mse

# Optimization Functions
'''
Gradient Descent:
 * Optimizes weight and bias by finding the minimum cost/loss of the bias and weight gradient
 * Subtracts the product of the learning rate and their individual gradient from their respective weight and biases
 * Some implementation sourced from GeeksForGeeks.com
'''
def gradientDescent(x, y, learnRate, stopThresh, maxIt):
    #initialize weight to 1, bias to 0 (will be optimized later)
    w = 1
    b = 0

    n = len(x) # number of elements in numpy array

    yPred = (w * x) + b # initialize prediction

    for i in range(maxIt):
        dw = -(2/n) * sum(x*(y-yPred)) # weight gradient
        db = -(2/n) * sum(y-yPred) # bias gradient

        #adjust weights and biases by subtracting the product of their respective gradients by the learning rate
        w -= learnRate * dw
        b -= learnRate * db

        yPred = (w * x) + b  # update predictions using the updated weights and biases
        if abs(dw) <= stopThresh and abs(db) <= stopThresh: # stop iterating once gradient of bias and weight less than threshold
            break
    if i == maxIt-1: # if max number of iterations reached, the model failed to reach the minimum threshold. warns the user to adjust learning rate and/or threshold to achieve more accurate result
        print('Max number of iterations reached. Increase the learning rate and/or threshold for more precise results')
    return w, b


x = np.array([1.2,2.4,2.5,4.1,3.2,4,6])
y = np.array([4.2,5.7,6.1,8.5,8.3,9,17])
weight, bias = gradientDescent(x, y, 0.01, 1e-6, 10000)

plt.scatter(x,y)
plt.plot(x,weight * x + bias,color='red')
plt.show()
