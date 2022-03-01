import turtle
import numpy as np
from sim import Env
#from net import Network
#from main import width, height, goal_dim, penalty_area_r

wn = turtle.Screen()
pen = turtle.Turtle()

atk = turtle.Turtle()
gk = turtle.Turtle()

def play(env: Env):

    wn.title("GKPositioning")
    wn.setup(width=env.width, height=env.height)
    wn.bgcolor("black")
    wn.tracer(0)
    wn.onscreenclick(lambda x, y: place_atk((x, y)))

    pen.speed('fastest')
    pen.pensize(2)

    atk.speed('fastest')
    atk.setpos((0, 0))
    atk.pensize(6)
    atk.pendown()
    atk.color('red')
    atk.dot()
    atk.penup()

    gk.speed('fastest')
    gk.setpos((0, env.center_goal.y))
    gk.pensize(6)
    gk.pendown()
    gk.color('green')
    gk.dot()
    gk.penup()

    # Bottom line
    pen.setpos(env.center_goal.get_tuple())
    pen.pendown()
    pen.color("white")
    pen.dot()
    pen.penup()

    # Goal
    pen.setpos((env.center_goal.x-env.goal_dim/2, env.center_goal.y))
    pen.pendown()
    pen.color("blue")
    pen.forward(env.goal_dim)
    pen.penup()

    # Penalty Area
    pen.setpos(env.center_goal.get_tuple())
    pen.color("white")
    pen.pendown()
    pen.circle(env.penalty_area_r)
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

    #place_gk(game(atk_pos, gk_net))

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
    env = Env()
    play(env)