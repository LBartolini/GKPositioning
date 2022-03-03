import numpy as np
from typing import List, Tuple
from sim import *
from net import *

EPOCHS_SIM = 10
POPULATION_SIM=50

EPOCH_TRAINING = 100
LR = 0.1

def one_hot(value, dimension) -> List:
    x = np.zeros(dimension+1, dtype=np.int8)
    x[int(value+((dimension+1)/2))] = 1

    return list(x)

def get_train_set(X, y, scores, env, top=0.3):
    treshold = sorted(scores)[-int(len(scores)*top)]

    X_train, yx_train, yy_train = [], [], []

    for i in range(len(scores)):
        if scores[i] >= treshold:
            x_one_hot, y_one_hot = get_one_hot_from_point(Point(X[i][0], X[i][1]), env)
            X_train.append([x_one_hot+y_one_hot])

            x_one_hot, y_one_hot = get_one_hot_from_point(Point(y[i][0], y[i][1]), env)
            yx_train.append([x_one_hot])
            yy_train.append([y_one_hot])

    return np.array(X_train), np.array(yx_train), np.array(yy_train), np.array(treshold)

def get_one_hot_from_point(point: Point, env: Env) -> Tuple[List, List]:
    x_one_hot = one_hot(point.x, env.width)
    y_one_hot = one_hot(point.y, env.height)

    return x_one_hot, y_one_hot

def create_net(inp_dim, out_dim, n_hidden=5) -> Network:
    net = Network()
    net.add(FCLayer(inp_dim, 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    
    for _ in range(n_hidden):
        net.add(FCLayer(100, 100))
        net.add(ActivationLayer(tanh, tanh_prime))

    net.add(FCLayer(100, out_dim))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.use(mse, mse_prime)

    return net

def get_gk_position(atk, net_x, net_y, env) -> Point:
    x_one_hot, y_one_hot = get_one_hot_from_point(atk, env)
    inp = np.array([x_one_hot+y_one_hot])

    pred_x = net_x.predict(inp)[0][0]
    pred_y = net_y.predict(inp)[0][0]

    return Point(np.argmax(pred_x)-env.width/2, np.argmax(pred_y)-env.height/2)

def main():
    env = Env()

    net_x = create_net(env.width+env.height+2, env.width+1)
    net_y = create_net(env.width+env.height+2, env.height+1)

    for k in range(EPOCHS_SIM):
        print(f"-----\nMAIN EPOCH: {k}\n")
        X = [] # atk_pos
        y = [] # gk_pos
        scores = [] # scores recorder for each iteration

        for i in range(POPULATION_SIM):
            atk = env.get_random_atk_position()
            gk = get_gk_position(atk, net_x, net_y, env)

            score = env.sim(atk, gk)

            X.append(atk.get_tuple())
            y.append(gk.get_tuple())
            scores.append(score)
        
        X_train, yx_train, yy_train, treshold = get_train_set(X, y, scores, env)
        print(f"Treshold: {treshold}\n\n")

        net_x.fit(X_train, yx_train, epochs=EPOCH_TRAINING, learning_rate=LR)
        net_y.fit(X_train, yy_train, epochs=EPOCH_TRAINING, learning_rate=LR)


if __name__ == '__main__':
    main()