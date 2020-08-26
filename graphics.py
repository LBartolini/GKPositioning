import turtle


def setup(width, height, goal_dim, penalty_area_r):
    wn = turtle.Screen()
    originx , originy = -width/2, height/2
    center_goal = -originy+20

    
    wn.title("GKPositioning")
    wn.setup(width=width, height=height)
    wn.bgcolor("black")
    wn.tracer(0)

    pen = turtle.Turtle()
    pen.speed(0)

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