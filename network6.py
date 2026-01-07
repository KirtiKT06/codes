"""network6.py
______________
An improved version of network.py, implementing the stochastic Gradient Descent learning algorithm for a FFNN.
Improvements include the addition of the cross-entropy cost function, L2 regularization, and better initiallization of network weights.
It uses GELU for inner layer activation function. It uses dropout as another regularization method.
It also uses softmax as an activation function for output layer instead of sigmoid.
It is not an optimized version.
"""
#### Libraries
# Standard Libraries
import random
# Third-party Library
import numpy as np

# The sigmoid function and derivative
def gelu(z):
    return 0.5 * z * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)
    ))
def gelu_prime(z):
    tanh_term = np.tanh(
        np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)
    )
    sech2 = 1 - tanh_term**2
    return 0.5 * (1 + tanh_term) + \
           0.5 * z * sech2 * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * z**2)

# The softmax function
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return exp_z/np.sum(exp_z)

#### Define the Quadratic and Cross-entropy cost functions
    
class CrossEntropyCost():
    @staticmethod
    def fn(a,y):
        return -np.sum(y * np.log(a + 1e-10))
    @staticmethod
    def delta(z,a,y):
        return (a-y)
    
#### Network Class

class network():
    def __init__(self, sizes, cost=CrossEntropyCost, dropout_rate = 0.5):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        self.dropout_rate = dropout_rate

    def default_weight_initializer(self):
        self.biases = []
        self.weights = []
        for y in self.sizes[1:]:
            self.biases.append(np.random.randn(y,1))
        for x,y in zip(self.sizes[:-1],self.sizes[1:]):
            self.weights.append(np.random.randn(y,x)*np.sqrt(2/x)) # For ReLU, the correct scaling is He initialization
    
    def feedforward(self,a, training = True):
        # Returns output of network of 'a' is input
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = gelu(np.dot(w, a) + b)
            if training:
                mask = (np.random.rand(*a.shape) < self.dropout_p) / self.dropout_p
                a = a * mask
        z_L = np.dot(self.weights[-1], a) + self.biases[-1]
        a_L = softmax(z_L)
        return a_L

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
        # Hidden layer
        for b,w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = gelu(z)
            activations.append(activation)
        # Output layer
        z_l = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z_l)
        activation = softmax(z_l)
        activations.append(activation)
        # backward pass
        delta = self.cost.delta(zs[-1], activations[-1],y)
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        # Fourth step is to back propagate to hidden layer
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = gelu_prime(z)
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
        test_results = [(np.argmax(self.feedforward(x, training = False)),y)
                        for (x,y) in test_data]
        return sum(int(x==y) for(x,y) in test_results)