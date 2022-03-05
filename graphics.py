import pygame
import time as t
from sim import *
from net import Network

RED = (255, 0, 0)
LIGHT_RED = (128, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


def graphic(net: Network, env: Env):
    window = pygame.display.set_mode((env.width, env.height))
    pygame.mouse.set_visible(False)

    while True:
        window.fill((0, 0, 0))
        pygame.draw.line(window, WHITE, env.goal_line_left.convert_cords(env), env.goal_line_right.convert_cords(env))
        pygame.draw.line(window, GREEN, env.left_post.convert_cords(env), env.right_post.convert_cords(env))
        pygame.draw.circle(window, GREEN, env.left_post.convert_cords(env), 2)
        pygame.draw.circle(window, GREEN, env.right_post.convert_cords(env), 2)
        pygame.draw.circle(window, GREEN, env.center_goal.convert_cords(env), env.penalty_area_r, 1)

        t.sleep(0.01)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if any(pygame.mouse.get_pressed()):
            atk = pygame.mouse.get_pos()
            pygame.draw.line(window, LIGHT_RED, atk, env.left_post.convert_cords(env))
            pygame.draw.line(window, LIGHT_RED, atk, env.right_post.convert_cords(env))
            pygame.draw.circle(window, RED, atk, 2)

            gk_x, gk_y = net.predict(np.array([[atk[0]/(env.width/2), atk[1]/(env.height/2)]]))[0][0]
            gk_x, gk_y = int(gk_x*(env.width//2)), int(gk_y*(env.height//2))
            gk = Point(gk_x, gk_y)

            pygame.draw.circle(window, GREEN, gk.convert_cords(env), 2)

        pygame.display.flip()


if __name__ == '__main__':
    env = Env(width=120, height=120)

    graphic(None, env)
