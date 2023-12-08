# Simple Neural Network Implementation
Spencer R. Karofsky

May 2023

I have recently been teaching myself neural networks. To gain a solid understanding of the fundamentals, 
I implemented a basic neural network that performs linear regression on a numpy array.

In my main program, I have implemented the following functions:

1) Activation Functions, which take an input and map it to a non-linear output.
	
	a) Sigmoid: $` y = \frac{1}{1+e^(-x)} `$

	* Sigmoid is used primarily for binary classification, where the output is between 0 and 1
	
	b) ReLU: $` y = max(0,x) `$

	* ReLU (Rectified Linear Unit) is a commonly used activation function because it is much faster than sigmoid
	
	c) Tanh:

	* Tanh (the hyperbolic tangent) takes in an input and returns a value between -1 and 1.

3) Loss Functions, which calculate the loss between the values the neural network predicts and the actual values
	
	a) The Mean Squared Error (MSE) is given by the formula: $` MSE = \frac{1}{n}  \Sigma^{n}_{i=1} (x_i - \hat{x}_i)^2 `$
	
 	* Mean squared error is a popular and efficient loss function

4) Optimizer Functions, which minimize the loss of the weights and biases in a neural network
	
	a) Gradient Descent
	
 	* Gradient descent works by starting with an initial weight and bias – commonly picked on a normal distribution with a mean of 0 and a standard deviation of 1. The gradient of the weight and bias losses is then calculated, and the product of the respective gradients and pre-specified learning rate is subtracted from the weight and bias. This is repeated over several iterations (epochs) until the loss is below a pre-specified threshold. The idea behind gradient descent is finding the minimum of the weight and bias functions with respect to the loss.


So far this neural network only works with linear regression; as a result, it never makes use of the activation functions. As I gain
a greater understanding of neural networks, I intend on adding quadratic and polynomial regression functionality, which will enable this
neural network to perform regression and classification on more complex datasets and problems.

<img width="626" alt="image" src="https://github.com/spencerkarofsky/neuralNetwork/assets/105813301/76d707ad-518b-4d37-9d9f-269160f39668">


I have really enjoyed working on this project, and I have significantly increased my understanding of neural networks in the process.
