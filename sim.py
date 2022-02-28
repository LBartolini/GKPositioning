import numpy as np
from net import Network

class Point:
    def __init__(self, x, y, is_null=False):
        self.x = x
        self.y = y
        self.is_null=is_null
    
    def get_tuple(self):
        return self.x, self.y

    def __repr__(self):
        return f'X: {self.x}, Y: {self.y}, is_null: {self.is_null}'
    

width, height = 600, 600
MAX_GK_DIM = 40
GOAL_LINE = 20

center_goal = Point(0, int(-height/2)+GOAL_LINE)
goal_dim = int(width*0.25)
gk_dim = min(int(goal_dim*0.5), MAX_GK_DIM)
penalty_area_r = int(width*0.25)

# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

def line(p1: Point, p2: Point):
    A = (p1.y - p2.y)
    B = (p2.x - p1.x)
    C = (p1.x*p2.y - p2.x*p1.y)
    return A, B, -C

def intersection(L1, L2) -> Point: 
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return Point(x, y)
    else:
        return Point(0, 0, True)

def distance(p1: Point, p2: Point):
    return np.sqrt(((p1.x-p2.x)**2)+((p1.y-p2.y)**2))

def get_inputs(atk: Point, center_goal: Point):
    # get polar coordinates of atk
    dist = distance(atk, center_goal)

    if atk.x != center_goal.x:
        angle_atk_goal = (atk.y-center_goal.y)/(atk.x-center_goal.x)
        angle = np.degrees(np.arctan(angle_atk_goal))
    else:
        angle = 90

    return dist/100, angle/100

def check_gk_pos(atk_left_line, atk_right_line, vertical_gk, gk: Point):
    int_left_ver: Point = intersection(atk_left_line, vertical_gk)
    int_right_ver: Point = intersection(atk_right_line, vertical_gk)

    if not int_left_ver.is_null and not int_right_ver.is_null:
        if int_left_ver.y <= gk.y <= int_right_ver.y or int_right_ver.y <= gk.y <= int_left_ver.y:
            return True
        else:
            return False
    else:
        return True

def compute_perp_atk_gk(atk_gk_line, gk: Point):
    try:
        slope_atk_gk_line = atk_gk_line[0]/atk_gk_line[1]  # slope = -a/b
        slope_perp = -1 / slope_atk_gk_line  # m
    except ZeroDivisionError:
        slope_perp = 0  # m

    try:
        int_perp = gk.y - (-slope_perp * gk.x)  # q
    except ZeroDivisionError:
        print("questo non dovrebbe mai accadere, wtf???")
        if gk.x == 0:
            int_perp = gk.y

    return slope_perp, 1, int_perp
    

def get_prob(center_goal: Point, atk: Point, gk: Point, gk_dim, goal_dim):
    prob = 1
    #gk = (r*np.cos(angle), r*np.sin(angle)+center_goal[1]) # convert polar to cartesian coordinates
    left_post = Point(center_goal.x-(goal_dim/2), center_goal.y)
    right_post = Point(center_goal.x+(goal_dim/2), center_goal.y)

    atk_gk_line = line(atk, gk)
    atk_left_line = line(atk, left_post)
    atk_right_line = line(atk, right_post)
    vertical_gk = line(gk, Point(gk.x,gk.y+1))

    if check_gk_pos(atk_left_line, atk_right_line, vertical_gk, gk):   
        perp_line = compute_perp_atk_gk(atk_gk_line, gk)

        left_int = intersection(atk_left_line, perp_line)
        right_int = intersection(atk_right_line, perp_line)

        #prob_left = (distance(left_int, gk) / (distance(left_int, gk) - gk_dim/2)) - 1
        #prob_right = (distance(right_int, gk) / (distance(right_int, gk) - gk_dim/2)) - 1
        prob_left = (max(distance(left_int, gk) - gk_dim/2, 0) / distance(left_int, gk))
        prob_right = (max(distance(right_int, gk) - gk_dim/2, 0) / distance(right_int, gk))

        prob = prob_left + prob_right

    return prob

def game(atk, net):
    inp = np.array(get_inputs(atk, center_goal))
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
        inp = np.array(get_inputs(atk, center_goal))
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

    inp = np.array(get_inputs((-10, 10), center_goal))
    pred = net.forward_propagation(inp)
    pred[0] *= penalty_area_r # distance to move
    pred[1] = np.degrees(pred[1]*2*np.pi)

    print(pred)
    '''
    atk = Point(10, 0)
    gk = Point(0, GOAL_LINE)

    print(get_prob(center_goal, atk, gk, gk_dim, goal_dim))
