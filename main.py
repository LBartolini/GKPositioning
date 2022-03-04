import numpy as np
from typing import List, Tuple
from sim import *
from net import *

import warnings
warnings.filterwarnings("ignore")

EPOCHS_SIM = 20
POPULATION_SIM = 5_000
ENV_STEPS = 50
REPS = 10 # repetitions of the same atk position with some randomness for gk position

RANDOMNESS = 12 # randomly chose in the range [-3, 3] and then summed to gk coords

EPOCH_TRAINING = 30
RESET_NET = False
LR = 0.01
DYN_LR = 1.1

HIDDEN_DIM = 50
HIDDEN_LAYERS = 2

WIDTH=60
HEIGHT=60

def one_hot(value, dimension) -> List:
    x = np.zeros(dimension+1, dtype=np.int8)
    x[int(value+((dimension+1)/2))] = 1

    return list(x)

def get_train_set(X, y, scores, env, top=0.2):
    scores_sorted, X_sorted, y_sorted  = list(zip(*sorted(zip(scores, X, y), reverse=True)))

    X_train, yx_train, yy_train = [], [], []

    for i in range(int(len(scores_sorted)*top)):
        if scores_sorted[i] == 0 or scores_sorted[i] < int(env.steps*(1-top*2)): continue

        x_one_hot, y_one_hot = get_one_hot_from_point(Point(X_sorted[i][0], X_sorted[i][1]), env)
        X_train.append([x_one_hot+y_one_hot])

        x_one_hot, y_one_hot = get_one_hot_from_point(Point(y_sorted[i][0], y_sorted[i][1]), env)
        yx_train.append([x_one_hot])
        yy_train.append([y_one_hot])

    return np.array(X_train), np.array(yx_train), np.array(yy_train),  np.mean(scores)/100, np.std(scores)/100, np.max(scores)/100

def get_one_hot_from_point(point: Point, env: Env) -> Tuple[List, List]:
    x_one_hot = one_hot(point.x, env.width)
    y_one_hot = one_hot(point.y, env.height)

    return x_one_hot, y_one_hot

def create_net(inp_dim, out_dim, n_hidden=2, hidden_dim=10) -> Network:
    net = Network()
    net.add(FCLayer(inp_dim, hidden_dim))
    net.add(ActivationLayer(tanh, tanh_prime))
    
    for _ in range(n_hidden):
        net.add(FCLayer(hidden_dim, hidden_dim))
        net.add(ActivationLayer(tanh, tanh_prime))

    net.add(FCLayer(hidden_dim, out_dim))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.use(mse, mse_prime)

    return net

def get_gk_position(atk, net_x, net_y, env) -> Point:
    x_one_hot, y_one_hot = get_one_hot_from_point(atk, env)
    inp = np.array([x_one_hot+y_one_hot])

    pred_x = net_x.predict(inp)[0][0]
    pred_y = net_y.predict(inp)[0][0]    

    return Point(np.argmax(pred_x)-env.width/2, np.argmax(pred_y)-env.height/2)

def randomize(gk, rnd) -> Point:
    offset_x = np.random.choice(rnd*2+1)-rnd
    offset_y = np.random.choice(rnd*2+1)-rnd

    return Point(gk.x+offset_x, gk.y+offset_y)

def main():
    lr = LR
    env = Env(width=WIDTH, height=HEIGHT, steps=ENV_STEPS*100)

    net_x = create_net(env.width+env.height+2, env.width+1, HIDDEN_LAYERS, HIDDEN_DIM)
    net_y = create_net(env.width+env.height+2, env.height+1, HIDDEN_LAYERS, HIDDEN_DIM)

    for k in range(EPOCHS_SIM):
        print(f"-----\nMAIN EPOCH: {k}\n")
        X = [] # atk_pos
        y = [] # gk_pos
        scores = [] # scores recorder for each iteration

        for _ in range(POPULATION_SIM):
            atk = env.get_random_atk_position()            
            gk_base = get_gk_position(atk, net_x, net_y, env)

            for _ in range(REPS):
                gk = randomize(gk_base, RANDOMNESS) 

                score = env.sim(atk, gk)

                X.append(atk.get_tuple())
                y.append(gk.get_tuple())
                scores.append(score)
        
        X_train, yx_train, yy_train, mean, std, max = get_train_set(X, y, scores, env, top=0.2)
        print(f"Mean: {mean} (std: {std})")
        print(f"Max: {max}\n\n")

        if len(X_train) == 0:
            print("JUMPED\n")
            continue
        
        if RESET_NET:
            net_x = create_net(env.width+env.height+2, env.width+1, HIDDEN_LAYERS, HIDDEN_DIM)
            net_y = create_net(env.width+env.height+2, env.height+1, HIDDEN_LAYERS, HIDDEN_DIM)

        print("Net X")
        net_x.fit(X_train, yx_train, epochs=EPOCH_TRAINING, learning_rate=lr)
        
        print("Net Y")
        net_y.fit(X_train, yy_train, epochs=EPOCH_TRAINING, learning_rate=lr)

        lr *= DYN_LR

        lr = min(lr, 0.1)
    
    print("Training Ended")
    while True:
        print('\n\n\n')
        x = int(input("Insert X: "))
        y = int(input("Insert Y: "))

        x_gk = net_x.predict(np.array([one_hot(x, env.width)+one_hot(y, env.height)]))
        y_gk = net_y.predict(np.array([one_hot(x, env.width)+one_hot(y, env.height)]))

        print((np.argmax(x_gk)-env.width/2, np.argmax(y_gk)-env.height/2))

if __name__ == '__main__':
    main()