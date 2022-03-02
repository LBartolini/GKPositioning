import numpy as np
from sim import *
from net import *

EPOCHS_SIM = 10
POPULATION_SIM=100

EPOCH_TRAINING = 3000
LR = 0.1

def get_train_set(X, y, scores, top=0.3):
    treshold = sorted(scores)[-int(len(scores)*top)]

    X_train, y_train = [], []

    for i in range(len(scores)):
        if scores[i] >= treshold:
            X_train.append([X[i]])
            y_train.append([y[i]])

    return np.array(X_train), np.array(y_train), treshold

def main():
    env = Env()

    net = Network()
    net.add(FCLayer(2, 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 2))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.use(mse, mse_prime)

    for k in range(EPOCHS_SIM):
        print(f"-----\nMAIN EPOCH: {k}\n")
        X = [] # atk_pos
        y = [] # gk_pos
        scores = [] # scores recorder for each iteration

        for i in range(POPULATION_SIM):
            atk, gk, score = env.sim(net)

            X.append(atk.get_tuple())
            y.append(gk.get_tuple())
            scores.append(score)
        
        X_train, y_train, treshold = get_train_set(X, y, scores)
        print(f"Treshold: {treshold}\n\n")

        net.fit(X_train, y_train, epochs=EPOCH_TRAINING, learning_rate=LR)


if __name__ == '__main__':
    main()