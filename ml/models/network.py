"""Network.py
--------------------
A module to implement the stocastic gradient descent learning algorithm for feedforward neural network.
Gradients are calculated using backpropagation"""
# Libraries
import random
import numpy as np
"""Next we need to calculate the sigmoid function and its derivative. We do not define it inside the class program.
We need to treat these functions as stand alone funct5ions rather than as a method of a class.
This is done to avoid calling the method at the end of the class. But one can definetly use that method. """
# The sigmoid function and derivative
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
"""We are now going to define the network class"""
# Define the Network Class
class network:
    def __init__(self, sizes):
        """The list 'sizes' contains thr number of neurons in the respective layers of the network.
        For example, if the list was [2,3,1], then it would be a three layered network with layer having 2 neurons, the second layer having 3 and the final layer having 1
        The biases and weights are initialized randomly, using a gausssian distribution with mean 0 and varience 1.
        We presume the first layer to be input layer, hence we do not set biases for those layers."""
        self.sizes = sizes
        self.num_layer = len(sizes)
        self.biases = []
        self.weights = []
        for y in sizes[1:]:
            self.biases.append(np.random.randn(y,1))
        for x,y in zip(sizes[:-1],sizes[1:]):
            self.weights.append(np.random.randn(y,x))
    def feedforward(self,a):
        # Returns output of network of 'a' is input
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Trains thr neural network using mini-batch stocastic gradient descent method.
        The training_data variable is a list of tuples (x,y) representing training input x and desired output y."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in range (0, n, mini_batch_size):
                batch = training_data[k:k+mini_batch_size]
                mini_batches.append(batch)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch{0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying SGD using backprop algorithm to a single mini batch.
        Here mini_batch is a list of tuples (x,y) and eta is the learning rate."""
        total_bias_grad = []
        total_weight_grad = []
        for b in self.biases:
            total_bias_grad.append(np.zeros(b.shape))
        for w in self.weights:
            total_weight_grad.append(np.zeros(w.shape))
        for x,y in mini_batch:
            db, dw = self.backprop(x,y)
            for i in range(len(self.weights)):
                total_bias_grad[i]+=db[i]
                total_weight_grad[i]+=dw[i]
        batch_size = len(mini_batch)
        for i in range(len(self.weights)):
            self.weights[i]-=(eta/batch_size)*total_weight_grad[i]
            self.biases[i]-=(eta/batch_size)*total_bias_grad[i]
    def backprop(self, x, y):
        # First step is to create empty gradient lists
        grad_b = []
        grad_w = []
        for w in self.weights:
            grad_w.append(np.zeros(w.shape))
        for b in self.biases:
            grad_b.append(np.zeros(b.shape))
        # Second step is the forward pass
        activation = x
        activations = [x] # A list to store all activations layer by layer
        zs = [] # A list to store all z vectors, layer by layer
        for i in range (len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Third step output layer error
        # Derivative of cost function with rspect to activation
        output_error = activations[-1]-y
        # Derivative of sigmoid
        sigmoid_prime = sigmoid(zs[-1])*(1-sigmoid(zs[-1]))
        delta = output_error*sigmoid_prime
        # Store the gradient  for the output layer
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        # Fourth step is to back propagate to hidden layer
        for layer in range(2, self.num_layer):
            z = zs[-layer]
            sigmoid_prime = sigmoid(z)*(1-sigmoid(z))
            # Error coming from next layer
            next_w = self.weights[-layer+1]
            delta = np.dot(next_w.transpose(),delta)*sigmoid_prime
            # Store the gradient
            grad_b[-layer] = delta
            grad_w[-layer] = np.dot(delta,activations[-layer-1].transpose())
        return grad_b, grad_w
    def forward_with_activations(self, x):
        """
        Performs a forward pass and RETURNS all activations.
        This does NOT affect training.
        """
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations, zs
    def evaluate (self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y)
                        for (x,y) in test_data]
        return sum(int(x==y) for(x,y) in test_results)
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x/
        \partial a for the output activation"""
        return (output_activations-y)