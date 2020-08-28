import turtle
import numpy as np
from sim import game
from net import Network
from main import HIDDEN_SIZE, width, height, goal_dim, penalty_area_r

wn = turtle.Screen()
pen = turtle.Turtle()
atk = turtle.Turtle()
gk = turtle.Turtle()

gk_net = Network(np.concatenate(([2], HIDDEN_SIZE, [2])), weights_path='saves/best_net.txt')

def play(width, height, goal_dim, penalty_area_r):
    originx , originy = -width/2, height/2
    center_goal = -originy+20

    wn.title("GKPositioning")
    wn.setup(width=width, height=height)
    wn.bgcolor("black")
    wn.tracer(0)
    wn.onscreenclick(lambda x, y: place_atk((x, y)))

    pen.speed(0)

    atk.speed('fastest')
    atk.setpos((0, 0))
    atk.pensize(6)
    atk.pendown()
    atk.color('red')
    atk.dot()
    atk.penup()

    gk.speed('fastest')
    gk.setpos((0, center_goal+30))
    gk.pensize(6)
    gk.pendown()
    gk.color('green')
    gk.dot()
    gk.penup()

    # Bottom line
    pen.setpos(originx, center_goal)
    pen.color("white")
    pen.pendown()
    pen.forward(width)
    pen.penup()

    # Goal
    pen.setpos(-goal_dim/2, center_goal)
    pen.color("blue")
    pen.pendown()
    pen.forward(goal_dim)
    pen.penup()

    # Penalty Area
    pen.setpos(0, center_goal-penalty_area_r)
    pen.color("white")
    pen.pendown()
    pen.circle(penalty_area_r)
    pen.penup()

    wn.mainloop()

def place_atk(atk_pos):
    atk.pendown()
    atk.color('black')
    atk.dot()
    atk.penup()

    atk.setpos(atk_pos)
    atk.pendown()
    atk.color('red')
    atk.dot()
    atk.penup()

    place_gk(game(atk_pos, gk_net))

def place_gk(gk_pos):
    gk.pendown()
    gk.color('black')
    gk.dot()
    gk.penup()

    gk.setpos(gk_pos)
    gk.pendown()
    gk.color('green')
    gk.dot()
    gk.penup()

if __name__ == '__main__':
    play(width, height, goal_dim, penalty_area_r)