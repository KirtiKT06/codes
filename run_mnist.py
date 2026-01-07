import mnist_loader
from network8 import network

def main():
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

    net = network([784, 100, 10])
    net.SGD(
        training_data=training_data,
        epochs=5,
        mini_batch_size=10,
        eta=0.01,
        lmbda=0.0,
        gamma=0.9,
        test_data=test_data,
        monitor_evaluation_accuracy = True
    )

if __name__ == "__main__":
    main()