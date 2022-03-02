import numpy as np

class Point:
    def __init__(self, x, y, is_null=False):
        self.x = x
        self.y = y
        self.is_null=is_null
    
    def get_tuple(self):
        return self.x, self.y

    def __repr__(self):
        return f'X: {self.x}, Y: {self.y}, is_null: {self.is_null}'

    @staticmethod
    def distance(p1, p2):
        return np.sqrt(((p1.x-p2.x)**2)+((p1.y-p2.y)**2))


class Line:
    def __init__(self, a=0, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c
    
    @staticmethod
    def from_points(p1: Point, p2: Point):
        a = (p1.y - p2.y)
        b = (p2.x - p1.x)
        c = -(p1.x*p2.y - p2.x*p1.y)

        return Line(a, b, c)
    
    def get_tuple(self):
        return self.a, self.b, self.c

    def __repr__(self):
        return f'A: {self.a}, B: {self.b}, C: {self.c}'

    @staticmethod
    def intersection(L1, L2) -> Point: 
        D  = L1.a * L2.b - L1.b * L2.a
        Dx = L1.c * L2.b - L1.b * L2.c
        Dy = L1.a * L2.c - L1.c * L2.a
        if D != 0:
            x = Dx / D
            y = Dy / D
            return Point(x, y)
        else:
            return Point(0, 0, True)



class Env:
    def __init__(self, width=600, height=800, GOAL_PERC=0.2, GK_WIDTH_PERC=0.4, PEN_AREA_PERC=0.3, MAX_GK_DIM=60, GOAL_LINE=10):
        self.width, self.height = width, height
        self.MAX_GK_DIM = MAX_GK_DIM
        self.GOAL_LINE = GOAL_LINE

        self.center_goal = Point(0, int(-height/2)+self.GOAL_LINE)
        self.goal_dim = int(width*GOAL_PERC)
        self.gk_dim = min(int(self.goal_dim*GK_WIDTH_PERC), self.MAX_GK_DIM) # diameter not radius
        self.penalty_area_r = int(width*PEN_AREA_PERC) # radius (180)

    def gaussian(self, x):
        a=25
        coeff = 2.5*a
        b=3/2*self.gk_dim
        return coeff/(a*np.sqrt(2*np.pi))*(np.exp(1)**(-0.5*(((x-b)/a)**2)))

    def atk_polar(self, atk: Point):
        # get polar coordinates of atk
        dist = Point.distance(atk, self.center_goal)

        if atk.x != self.center_goal.x:
            angle_atk_goal = (atk.y-self.center_goal.y)/(atk.x-self.center_goal.x)
            angle = np.degrees(np.arctan(angle_atk_goal))
        else:
            angle = 90

        return dist/100, angle/100

    def compute_perp_atk_gk(self, atk_gk_line: Line, gk: Point) -> Line:
        try:
            slope_atk_gk_line = atk_gk_line.a/atk_gk_line.b  # slope = -a/b
            slope_perp = -1 / slope_atk_gk_line  # m
        except ZeroDivisionError:
            slope_perp = 0  # m

        try:
            int_perp = gk.y - (-slope_perp * gk.x)  # q
        except ZeroDivisionError:
            print("questo non dovrebbe mai accadere, wtf???")
            if gk.x == 0:
                int_perp = gk.y

        return Line(slope_perp, 1, int_perp)
    

    def get_prob(self, atk: Point, gk: Point, lob=False):
        probs = []

        #gk = (r*np.cos(angle), r*np.sin(angle)+center_goal[1]) # convert polar to cartesian coordinates
        left_post = Point(self.center_goal.x-(self.goal_dim/2), self.center_goal.y)
        right_post = Point(self.center_goal.x+(self.goal_dim/2), self.center_goal.y)

        atk_gk_line = Line.from_points(atk, gk)
        atk_left_line = Line.from_points(atk, left_post)
        atk_right_line = Line.from_points(atk, right_post)
    
        perp_line = self.compute_perp_atk_gk(atk_gk_line, gk)

        left_int = Line.intersection(atk_left_line, perp_line)
        right_int = Line.intersection(atk_right_line, perp_line)

        #computing probs of atk scoring based on the position of the goalie and its width
        prob_left = (max(Point.distance(left_int, gk) -
                        (self.gk_dim/2), 0) / Point.distance(left_int, gk))
        prob_right = (max(Point.distance(right_int, gk) -
                        (self.gk_dim/2), 0) / Point.distance(right_int, gk))

        probs.append((prob_left, 0.4))
        probs.append((prob_right, 0.4))

        if lob:
            #computing lob probability
            dst_gk_goal = Point.distance(gk, self.center_goal)
            dst_gk_atk = Point.distance(gk, atk)

            val_dst_atk_gk = self.gaussian(dst_gk_atk)
            val_dst_gk_goal = max((dst_gk_goal / (self.gk_dim/2))-1, 0)

            prob_lob = val_dst_gk_goal * val_dst_atk_gk

            probs.append((prob_lob, 0.2))


        #calculating final prob
        prob = min(sum([x[0]*x[1] for x in probs]), 1)

        return prob

    def get_gk_position(self, atk: Point, net) -> Point:
        inp = np.array([self.atk_polar(atk)])
        pred = net.predict(inp)[0][0]
        pred[0] *= self.penalty_area_r # distance to move (aka r)
        pred[1] = np.degrees(pred[1]*np.pi) # direction where to move from the center of the goal (aka angle)
        
        # convert polar to cartesian coordinates
        gk_pos = Point(pred[0]*np.cos(pred[1]), self.center_goal.y+pred[0]*np.sin(pred[1])) 

        return gk_pos

    def sim(self, net, steps=50):
        #returns the score obtained by the agent on a given atk position
        score = 0

        # random select spot for attacker
        x = np.random.randint(-self.width/2, self.width/2)
        y = min(np.random.randint(-self.height/2, self.height/2)+self.GOAL_LINE, self.height/2)
        atk = Point(x, y)

        # ask the network where to position the goalkeeper
        gk_pos = self.get_gk_position(atk, net)

        # compute probability of scoring
        prob_goal = self.get_prob(atk, gk_pos)*1000 # 1_000 for accuracy

        # iterate over steps
        for _ in range(steps):
            # store if GK took goal or not
            if np.random.randint(0, 1000) > prob_goal:
                score += 1
               
        return atk, gk_pos, score

if __name__ == '__main__':
    env = Env()

    atk = Point(0, -370)
    gk = Point(env.center_goal.x, env.center_goal.y+50)

    #goal = -490
    #top_pen_area = -310

    print(env.get_prob(atk, gk, lob=True))
