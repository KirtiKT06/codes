"""network2.py
______________
An improved version of network.py, implementing the stochastic Gradient Descent learning algorithm for a FFNN.
Improvements include the addition of the cross-entropy cost function, rehularization, and better initiallization of network weights.
It is not an optimized version.
"""
#### Libraries
# Standard Libraries
import random
# Third-party Library
import numpy as np

# The sigmoid function and derivative
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#### Define the Quadratic and Cross-entropy cost functions

class QuadraticCost():
    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2
    @staticmethod
    def delta(z,a,y):
        return (a-y)*sigmoid_prime(z)
    
class CrossEntropyCost():
    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    @staticmethod
    def delta(z,a,y):
        return (a-y)
    
#### Network Class

class network():
    def __init__(self, sizes, cost=QuadraticCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = []
        self.weights = []
        for y in self.sizes[1:]:
            self.biases.append(np.random.randn(y,1))
        for x,y in zip(self.sizes[:-1],self.sizes[1:]):
            self.weights.append(np.random.randn(y,x)/np.sqrt(x))
    
    def large_weight_initializer(self):
        self.biases = []
        self.weights = []
        for y in self.sizes[1:]:
            self.biases.append(np.random.randn(y,1))
        for x,y in zip(self.sizes[:-1],self.sizes[1:]):
            self.weights.append(np.random.randn(y,x))
    
    def feedforward(self,a):
        # Returns output of network of 'a' is input
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
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
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1],y)
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        # Fourth step is to back propagate to hidden layer
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid(z)*(1-sigmoid(z))
            # Error coming from next layer
            next_w = self.weights[-layer+1]
            delta = np.dot(next_w.transpose(),delta)*sp
            # Store the gradient
            grad_b[-layer] = delta
            grad_w[-layer] = np.dot(delta,activations[-layer-1].transpose())
        return grad_b, grad_w
    
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
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
            self.weights[i] = (
                (1-eta*(lmbda/n))*self.weights[i]
                -(eta/len(mini_batch))*total_weight_grad[i]
                )
            self.biases[i] = (
                self.biases[i]
                -(eta/len(mini_batch))*total_bias_grad[i]
                )

    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            lmbda = 0.0, 
            test_data = None, 
            monitor_evaluation_accuracy = False):
        if test_data is not None:
            n_data = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in range (0, n, mini_batch_size):
                batch = training_data[k:k+mini_batch_size]
                mini_batches.append(batch)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            if monitor_evaluation_accuracy:
                accuracy = self.evaluate(test_data)
                print("Epoch{0}:{1}/{2}".format(j, accuracy, n_data))
            else:
                print("Epoch {0} complete".format(j))
    def evaluate (self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y)
                        for (x,y) in test_data]
        return sum(int(x==y) for(x,y) in test_results)