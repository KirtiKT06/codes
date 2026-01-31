import numpy as np
from ml.models.network import network
# 1. training the neural netwok to learn the XOR gate
training_data = [(np.array([[0],[0]]), np.array([[0]])),
                 (np.array([[0],[1]]), np.array([[1]])), 
                 (np.array([[1],[0]]), np.array([[1]])), 
                 (np.array([[1],[1]]), np.array([[0]]))]
# 2. Create network
net = network([2,3,1])

# 3. train the network
net.SGD(training_data, epochs=5000, mini_batch_size=4, eta=3.0)

#4: Test results
print("testing XOR gate:")
print(" 0 XOR 0 = ", net.feedforward(np.array([[0],[0]])))
print(" 0 XOR 1 = ", net.feedforward(np.array([[0],[1]])))
print(" 1 XOR 0 = ", net.feedforward(np.array([[1],[0]])))
print(" 1 XOR 1 = ", net.feedforward(np.array([[1],[1]])))