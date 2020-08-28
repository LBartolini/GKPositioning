import numpy as np
from net import Network

width, height = 600, 600
MAX_GK_DIM = 80

center_goal = (0, int(-height/2)+20)
goal_dim = int(width*0.25)
gk_dim = min(int(goal_dim*0.5), MAX_GK_DIM)
penalty_area_r = int(width*0.25)

# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

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

def check_gk_pos(atk_left_line, atk_right_line, vertical_gk, gk):
    int_left_ver = intersection(atk_left_line, vertical_gk)
    int_right_ver = intersection(atk_right_line, vertical_gk)

    if not int_left_ver and not int_right_ver:
        if int_left_ver[1] <= gk[1] <= int_right_ver[1] or int_right_ver[1] <= gk[1] <= int_left_ver[1]:
            return True
        else:
            return False
    else:
        return True
    

def get_prob(center_goal, atk, gk, gk_dim, goal_dim):
    prob = 1
    #gk = (r*np.cos(angle), r*np.sin(angle)+center_goal[1]) # convert polar to cartesian coordinates
    left_goal = center_goal[0]-(goal_dim/2), center_goal[1]
    right_goal = center_goal[0]+(goal_dim/2), center_goal[1]

    atk_gk_line = line(atk, gk)
    atk_left_line = line(atk, left_goal)
    atk_right_line = line(atk, right_goal)
    vertical_gk = line(gk, (gk[0],gk[1]+1))

    if check_gk_pos(atk_left_line, atk_right_line, vertical_gk, gk):
        try:
            slope_atk_gk_line = atk_gk_line[0]/atk_gk_line[1] # slope = -a/b
        except ZeroDivisionError:
            slope_atk_gk_line = 10e6

        slope_perp = -1 / slope_atk_gk_line # m
        try:
            int_perp = gk[1] - (-slope_perp * gk[0]) # q
        except ZeroDivisionError:
            if gk[0] == 0:
                int_perp = gk[1]

        perp_line = (slope_perp, 1, int_perp)
        left_int = intersection(atk_left_line, perp_line)
        right_int = intersection(atk_right_line, perp_line)

        #prob_left = (distance(left_int, gk) / (distance(left_int, gk) - gk_dim/2)) - 1
        #prob_right = (distance(right_int, gk) / (distance(right_int, gk) - gk_dim/2)) - 1
        prob_left = (max(distance(left_int, gk) - gk_dim/2, 0) / distance(left_int, gk))
        prob_right = (max(distance(right_int, gk) - gk_dim/2, 0) / distance(right_int, gk))

        prob = prob_left + prob_right

    return prob

def game(atk, net):
    inp = np.array(get_inputs(atk, center_goal, goal_dim))
    pred = net.forward_propagation(inp)
    pred[0] *= penalty_area_r # distance to move (aka r)
    pred[1] = np.degrees(pred[1]*2*np.pi) # direction where to move from the center of the goal (aka angle)
    gk_pos = (pred[0]*np.cos(pred[1]), pred[0]*np.sin(pred[1])+center_goal[1]) # convert polar to cartesian coordinates

    return gk_pos

def simulation(net, rounds=100):
    score = 0

    # iterate over rounds
    for _ in range(rounds):
        # random select spot for attacker
        atk = center_goal # to get true the while contition
        while distance(center_goal, atk) <= penalty_area_r/2: # / 2 because if the attacker get too close the GK simply gets the ball
            atk = np.random.randint(-width/2, width/2, size=2)
        
        # ask the network where to position
        inp = np.array(get_inputs(atk, center_goal, goal_dim))
        pred = net.forward_propagation(inp)
        pred[0] *= penalty_area_r # distance to move (aka r)
        pred[1] = np.degrees(pred[1]*2*np.pi) # direction where to move from the center of the goal (aka angle)
        gk_pos = (pred[0]*np.cos(pred[1]), pred[0]*np.sin(pred[1])+center_goal[1]) # convert polar to cartesian coordinates
        
        # select where to shoot
        prob_goal = get_prob(center_goal, atk, gk_pos, gk_dim, goal_dim)*100

        # store if GK took goal or not
        if np.random.randint(0, 100) <= prob_goal:
            continue
        else:
            score += 1
            
    return score

if __name__ == '__main__':
    #setup(width, height, goal_dim, penalty_area_r)
    '''
    net = Network(np.concatenate(([2], [10], [2])))        

    inp = np.array(get_inputs((-10, 10), center_goal, goal_dim))
    pred = net.forward_propagation(inp)
    pred[0] *= penalty_area_r # distance to move
    pred[1] = np.degrees(pred[1]*2*np.pi)

    print(pred)
    '''
    atk = (-200, -100)
    gk_ = (-100, -200)
    print(get_prob(center_goal, atk, gk_, gk_dim, goal_dim))