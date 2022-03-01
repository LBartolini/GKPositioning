import numpy as np
from sim import *
from net import *

EPOCHS = 500

def main():
    # training data
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    net = Network()
    net.add(FCLayer(2, 10))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(10, 10))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(10, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    # train
    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

    # test
    out = net.predict(x_train)
    print(out)


if __name__ == '__main__':
    main()