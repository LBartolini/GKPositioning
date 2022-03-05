import math
import numpy as np


class Point:
    def __init__(self, x, y, is_null=False):
        self.x = x
        self.y = y
        self.is_null = is_null

    def get_tuple(self):
        return self.x, self.y

    def convert_cords(self, env):
        return (self.x+(env.width//2), abs(self.y-(env.height//2)))

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
    def intersection(l1, l2) -> Point:
        d = l1.a * l2.b - l1.b * l2.a
        Dx = l1.c * l2.b - l1.b * l2.c
        Dy = l1.a * l2.c - l1.c * l2.a

        if d != 0:
            x = Dx / d
            y = Dy / d

            return Point(x, y)
        else:
            return Point(0, 0, True)


class Env:
    def __init__(self, width=600, height=600, steps=50, GOAL_PERC=0.2, GK_WIDTH_PERC=0.6, PEN_AREA_PERC=0.3, MAX_GK_DIM=60, GOAL_LINE=5):
        self.width, self.height = width, height
        self.MAX_GK_DIM = MAX_GK_DIM
        self.GOAL_LINE = GOAL_LINE
        self.steps = steps

        self.center_goal = Point(0, int(-height/2)+self.GOAL_LINE)
        self.goal_dim = int(width*GOAL_PERC)
        self.gk_dim = min(int(self.goal_dim*GK_WIDTH_PERC),
                          self.MAX_GK_DIM)  # diameter not radius
        self.penalty_area_r = int(width*PEN_AREA_PERC)

        self.goal_line_left = Point(
            self.center_goal.x-self.width//2, self.center_goal.y)
        self.goal_line_right = Point(
            self.center_goal.x+self.width//2, self.center_goal.y)

        self.left_post = Point(self.center_goal.x-self.goal_dim//2, self.center_goal.y)
        self.right_post = Point(self.center_goal.x+self.goal_dim//2, self.center_goal.y)

    def gaussian(self, x):
        a = 25
        coeff = 2.5*a
        b = 3/2*self.gk_dim
        return coeff/(a*np.sqrt(2*np.pi))*(np.exp(1)**(-0.5*(((x-b)/a)**2)))

    def compute_perp_atk_gk(self, atk_gk_line: Line, gk: Point) -> Line:
        try:
            slope_atk_gk_line = - atk_gk_line.a / atk_gk_line.b  # slope = -a/b
            slope_perp = -1 / slope_atk_gk_line  # m
        except ZeroDivisionError:
            slope_perp = 0  # m

        int_perp = gk.y - (-slope_perp * gk.x)  # q

        return Line(slope_perp, 1, int_perp)

    def check_gk_position(self, gk: Point) -> bool:
        # gk need to be inside penalty area
        if Point.distance(gk, self.center_goal) > self.penalty_area_r:
            return True

        # gk has to be not behind the goal line
        if gk.y < self.center_goal.y:
            return True

        # gk needs to be inside the field
        if gk.x > self.width/2 or gk.x < -self.width/2:
            return True
        if gk.y > self.height/2 or gk.y < -self.height/2:
            return True

        return False

    def get_prob(self, atk: Point, gk: Point, lob=True):
        if self.check_gk_position(gk):
            return 1

        probs = []
        remaining_prob = 0.8 if lob else 1

        # gk = (r*np.cos(angle), r*np.sin(angle)+center_goal[1]) # convert polar to cartesian coordinates

        atk_left_line = Line.from_points(atk, self.left_post)
        atk_right_line = Line.from_points(atk, self.right_post)

        atk_gk_line = Line.from_points(atk, gk)
        perp_line = self.compute_perp_atk_gk(atk_gk_line, gk)

        left_int = Line.intersection(atk_left_line, perp_line)
        right_int = Line.intersection(atk_right_line, perp_line)

        # computing probs of atk scoring based on the position of the goalie and its width
        prob_left = (max(Point.distance(left_int, gk) -
                         (self.gk_dim/2), 0) / Point.distance(left_int, gk))
        prob_right = (max(Point.distance(right_int, gk) -
                          (self.gk_dim/2), 0) / Point.distance(right_int, gk))

        probs.append((prob_left, remaining_prob/2))
        probs.append((prob_right, remaining_prob/2))

        if lob:
            # computing lob probability
            dst_gk_goal = Point.distance(gk, self.center_goal)
            dst_gk_atk = Point.distance(gk, atk)

            val_dst_atk_gk = self.gaussian(dst_gk_atk)
            val_dst_gk_goal = max((dst_gk_goal / (self.gk_dim/2))-1, 0)

            prob_lob = val_dst_gk_goal * val_dst_atk_gk

            probs.append((prob_lob, 0.2))

        # calculating final prob
        prob = min(sum([x[0]*x[1] for x in probs]), 1)

        return prob

    def get_random_atk_position(self):
        x = np.random.randint(-self.width/2, self.width/2)
        y = min(np.random.randint(-self.height/2, self.height/2) +
                self.GOAL_LINE, self.height/2)

        while Point.distance(Point(x, y), self.center_goal) < self.penalty_area_r:
            y = min(np.random.randint(-self.height/2, self.height/2) +
                    self.GOAL_LINE, self.height/2)

        return Point(x, y)

    def sim(self, atk, gk_pos):
        # returns the score obtained by the agent on a given atk position

        # compute probability of scoring
        prob_goal = self.get_prob(atk, gk_pos, lob=False)

        if math.isnan(prob_goal):
            return 0

        return int((1-prob_goal)*self.steps)


if __name__ == '__main__':
    env = Env(width=80, height=80)

    atk = Point(0, 0)
    gk = Point(0, -29)

    print(env.get_prob(atk, gk, lob=False))
