"""
Simple Neural Network Implementation
Spencer R. Karofsky
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    # Activation Functions
    def sigmoid(self, x):  # 1/1+e^-x
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        y = np.tanh(x)
        return y

    # Loss Functions
    '''
    Mean Squared Error:
     * Average of the sum of the squares of the differences (residuals) between expected and predicted values
     * mean squared error = (SUM(predicted-target)^2)/n
     * Implementation sourced from GeeksForGeeks.com
    '''

    def mse(self, targets, preds):  # mean squared error
        mse = np.square(np.subtract(preds, targets)).mean()
        return mse

    # Optimization Functions

    '''
    Gradient Descent:
     * Optimizes weight and bias by finding the minimum cost/loss of the bias and weight gradient
     * Subtracts the product of the learning rate and their individual gradient from their respective weight and biases
     * Some implementation sourced from GeeksForGeeks.com
    '''
    def gradientDescent(self, x, y, w, b, learnRate, stopThresh, maxIt):
        # calculate loss
        # For graphing loss function later on
        weightLoss = np.array([])
        biasLoss = np.array([])

        n = len(x)  # number of elements in numpy array

        yPred = (w * x) + b  # initialize prediction

        for i in range(maxIt):
            dw = -(2 / n) * sum(x * (y - yPred))  # weight gradient
            db = -(2 / n) * sum(y - yPred)  # bias gradient

            # adjust weights and biases by subtracting the product of their respective gradients by the learning rate
            w -= learnRate * dw
            b -= learnRate * db

            yPred = (w * x) + b  # update predictions using the updated weights and biases

            # use loss function (mean squared error) to compute the loss for weight and bias
            # the goal is to minimize the weight and bias gradients, so their gradients are compared to 0
            weightMSE = self.mse(0, dw)
            biasMSE = self.mse(0, db)

            weightLoss = np.append(weightLoss, weightMSE)
            biasLoss = np.append(biasLoss, biasMSE)

            if abs(dw) <= stopThresh and abs(
                    db) <= stopThresh:  # stop iterating once gradient of bias and weight less than threshold
                break
        if i == maxIt - 1:  # if max number of iterations reached, the model failed to reach the minimum threshold. warns the user to adjust learning rate and/or threshold to achieve more accurate result
            print(
                'Max number of iterations reached. Increase the learning rate and/or threshold for more precise results')
        else:
            print(f'Gradient descent completed in {i + 1} of {maxIt} iterations.')
            print(f'Weight accurate to {dw}')
            print(f'Bias accurate to {db}')
        return w, b, weightLoss, biasLoss

    # Display functions
    def displayLoss(self, weightLoss, biasLoss):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(weightLoss, color='red', )
        ax1.set_ylabel('Weight Loss')
        ax2.plot(biasLoss, color='blue')
        ax2.set_ylabel('Bias Loss')
        ax2.set_xlabel('Epoch')
        plt.show()

    def displayLineFit(self, x, y, weight, bias):
        plt.title('Goodness of Fit Optimized by Gradient Descent')
        plt.scatter(x, y)
        plt.plot(x, weight * x + bias, color='red')
        plt.show()


'''
#quadratic relationship between x and y
x = np.array([1.2, 2.1, 2.9, 4.1, 4.4, 5.4, 6])
y = np.array([4.2, 4.7, 5.5, 8.3, 8.6, 15, 27])
'''

x = np.linspace(0, 10, 50)
y = np.multiply(x, 3) + 4
yVar = np.random.normal(0, 3, size=50)
y = np.add(y, yVar)

# initialize weight and biases to random values (will be adjusted in optimizer function)
w = np.random.normal(0, 1, size=1)
b = np.random.normal(0, 1, size=1)

nn = NeuralNetwork()
w, b, wL, bL = nn.gradientDescent(x, y, w, b, 0.01, 1e-6, 10000)
nn.displayLineFit(x, y, w, b)
nn.displayLoss(wL, bL)
