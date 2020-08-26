import numpy as np
from net import Network
from graphics import *

width, height = 600, 600
MAX_GK_DIM = 80

center_goal = (0, (-height/2)+20)
goal_dim = int(width*0.25)
gk_dim = min(int(goal_dim*0.5), MAX_GK_DIM)
penalty_area_r = int(width*0.25)

def distance(p1, p2):
    return np.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def get_inputs(atk, center_goal, goal_dim):
    dist = distance(atk, center_goal)

    if atk[0] != center_goal[0]:
        angle_atk_goal = (atk[1]-center_goal[1])/(atk[0]-center_goal[0])
    else:
        angle_atk_goal = 10e6
    angle = np.arctan(angle_atk_goal)

    return dist/100, np.degrees(angle)/100

def simulation(net, rounds=100):
    score = 0

    # iterate over rounds
    for _ in rounds:
        # random select spot where to shoot
        atk = center_goal # to get true the while contition
        while distance(center_goal, atk) <= penalty_area_r/2: # / 2 because if the attacker get too close the GK simply gets the ball
            atk = np.randint(-width/2, size=2)
        
        # ask the network where to position
        inp = np.array(get_inputs(atk, center_goal, goal_dim))
        pred = net.forward_propagation(inp)
        pred[0] *= penalty_area_r # distance to move
        pred[1] = np.degrees(pred[1]*2*np.pi) # direction where to move from the center of the goal

    # select where to shoot
    # store if GK took goal or not
    
    return score

if __name__ == '__main__':
    #setup(width, height, goal_dim, penalty_area_r)
    net = Network(np.concatenate(([2], [10], [2])))        

    inp = np.array(get_inputs((-10, 10), center_goal, goal_dim))
    pred = net.forward_propagation(inp)
    pred[0] *= penalty_area_r # distance to move
    pred[1] = np.degrees(pred[1]*2*np.pi)

    print(pred)