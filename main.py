from graphics import graphic
import numpy as np
from sim import *
from net import *
from graphics import graphic

import warnings
warnings.filterwarnings("ignore")

EPOCHS_SIM = 5
POPULATION_SIM = 5_000
ENV_STEPS = 50
REPS = 20 # repetitions of the same atk position with some randomness for gk position

RANDOMNESS = 15 # randomly chose in the range [-3, 3] and then summed to gk coords

EPOCH_TRAINING = 50
RESET_NET = False
LR = 0.01
MAX_LR = 0.1
DYN_LR = 1.1 # set to 1 for static_lr

HIDDEN_DIM = 20
HIDDEN_LAYERS = 10

WIDTH=100
HEIGHT=100

def get_train_set(X, y, scores, env: Env, top=0.2):
    scores_sorted, X_sorted, y_sorted  = list(zip(*sorted(zip(scores, X, y), reverse=True)))
    X_train, y_train = [], []

    for i in range(int(len(scores_sorted)*top)):
        if scores_sorted[i] == 0 or scores_sorted[i] < int(env.steps*(1-top*2)): continue

        X_train.append([[X_sorted[i][0]/(env.width/2), X_sorted[i][1]/(env.height/2)]])
        y_train.append([[y_sorted[i][0]/(env.width/2), y_sorted[i][1]/(env.height/2)]])

    return np.array(X_train),  np.array(y_train),  np.mean(scores)/100, np.std(scores)/100, np.max(scores)/100

def create_net(inp_dim, out_dim, n_hidden=2, hidden_dim=10) -> Network:
    net = Network()
    net.add(FCLayer(inp_dim, hidden_dim))
    net.add(ActivationLayer(tanh, tanh_prime))
    
    for _ in range(n_hidden):
        net.add(FCLayer(hidden_dim, hidden_dim))
        net.add(ActivationLayer(tanh, tanh_prime))

    net.add(FCLayer(hidden_dim, out_dim))
    net.add(ActivationLayer(tanh, tanh_prime))

    net.use(mse, mse_prime)

    return net

def get_gk_position(atk: Point, net: Network, env: Env) -> Point:
    x, y = atk.get_tuple()
    x /= (env.width/2)
    y /= (env.height/2)

    inp = np.array([[x, y]])

    pred_x, pred_y = net.predict(inp)[0][0]

    return Point(int(pred_x*(env.width/2)), int(pred_y*(env.height/2)))

def randomize(gk: Point, rnd) -> Point:
    offset_x = np.random.choice(rnd*2+1)-rnd
    offset_y = np.random.choice(rnd*2+1)-rnd

    return Point(gk.x+offset_x, gk.y+offset_y)

def main():
    lr = LR
    env = Env(width=WIDTH, height=HEIGHT, steps=ENV_STEPS*100)

    net = create_net(2, 2, HIDDEN_LAYERS, HIDDEN_DIM)

    for k in range(EPOCHS_SIM):
        print(f"-----\nMAIN EPOCH: {k}\n")
        X = [] # atk_pos
        y = [] # gk_pos
        scores = [] # scores recorder for each iteration

        for _ in range(POPULATION_SIM):
            atk = env.get_random_atk_position()            
            gk_base = get_gk_position(atk, net, env)

            for _ in range(REPS):
                gk = randomize(gk_base, RANDOMNESS) 

                score = env.sim(atk, gk)

                X.append(atk.get_tuple())
                y.append(gk.get_tuple())
                scores.append(score)
        
        X_train, y_train, mean, std, max = get_train_set(X, y, scores, env, top=0.2)
        print(f"Mean: {mean} (std: {std})")
        print(f"Max: {max}\n\n")

        if len(X_train) == 0:
            print("JUMPED\n")
            continue
        
        if RESET_NET:
            net = create_net(2, 2, HIDDEN_LAYERS, HIDDEN_DIM)
            
        net.fit(X_train, y_train, epochs=EPOCH_TRAINING, learning_rate=lr)

        lr *= DYN_LR
        lr = min(lr, MAX_LR)
    
    print("\nTraining Ended")

    graphic(net, env)

    '''
    while True:
        print('\n\n')
        x = int(input("Insert X: "))
        y = int(input("Insert Y: "))

        gk_x, gk_y = net.predict(np.array([[x/(env.width/2), y/(env.height/2)]]))[0][0]

        print(int(gk_x*(env.width/2)), int(gk_y*(env.height/2)))
    '''

if __name__ == '__main__':
    main()