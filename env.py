import numpy as np
from numpy.random import *
from math import *
import pygame
from pygame.locals import *
import sys

class Environment():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((500, 600))
        self.food = randint(0,501,(2,15,2))
        self.agent = np.array([rand()*470+10,rand()*470+10, randint(0,360), 0., 0.])
        self.last_D = 0.
        self.timestep = 0
        clock = pygame.time.Clock()
        clock.tick(200)

    def reset(self):
        self.food = randint(0,501,(2,15,2))
        self.agent = np.array([rand()*470+10,rand()*470+10, randint(0,360), 0., 0.])
        self.last_D = 0.
        self.timestep = 0
        return self.state()

    def eat(self):
        ax = self.agent[0]
        ay = self.agent[1]
        for i in xrange(2):
            for j in xrange(15):
                fx = self.food[i][j][0]
                fy = self.food[i][j][1]
                if (ax-fx)**2 + (ay-fy)**2 <= 30**2:
                    self.agent[3+i] += 1.
                    self.food[i][j][0] = randint(0,501)
                    self.food[i][j][1] = randint(0,501)

    def state(self):
        state = np.array([30. for i in range(2+19*3)])
        state[0] = self.agent[3]
        state[1] = self.agent[4]
        #state[2] = self.agent[3]
        #state[3] = self.agent[4]
        x = self.agent[0]
        y = self.agent[1]
        p = self.agent[2]
        for i in xrange(19):
            dx = cos(radians(p-90+10*i))*150*0.01
            dy = sin(radians(p-90+10*i))*150*0.01
            for l in xrange(15):
                fx0 = self.food[0][l][0]
                fy0 = self.food[0][l][1]
                fx1 = self.food[1][l][0]
                fy1 = self.food[1][l][1]
                for j in xrange(11):
                    if x+dx*j<0. or x+dx*j>500. or y+dy*j<0. or y+dy*j>500:
                        if state[2+i] == 30:
                            state[2+i] == 30.*j*0.1
                    if (x+dx*j-fx0)**2 + (y+dy*j-fy0)**2 <= 10**2:
                        if state[2+19+i] == 30.:
                            state[2+19+i] = 30.*j*0.1
                    if (x+dx*j-fx1)**2 + (y+dy*j-fy1)**2 <= 10**2:
                        if state[2+19+19+i] == 30.:
                            state[2+19+19+i] = 30.*j*0.1
        return state

    def render(self):
        self.screen.fill((184,224,210))
        for i in xrange(15):
            pygame.draw.circle(self.screen, (231, 76, 60),(self.food[0][i][0],self.food[0][i][1]), 10)
            pygame.draw.circle(self.screen, (52, 152, 219),(self.food[1][i][0],self.food[1][i][1]), 10)
        for i in xrange(19):
            x = self.agent[0]
            y = self.agent[1]
            p = self.agent[2]
            pygame.draw.line(self.screen, (0,0,0), (int(x),int(y)), (int(x+cos(radians(p-90+10*i))*150),int(y+sin(radians(p-90+10*i))*150)), 1)
        pygame.draw.circle(self.screen, (149, 165, 166),(int(self.agent[0]),int(self.agent[1])), 20)
        pygame.draw.rect(self.screen, (231, 76, 60), Rect(0,500,int(500*(self.agent[3]+10.)/20.),50))
        pygame.draw.rect(self.screen, (52, 152, 219), Rect(0,550,int(500*(self.agent[4]+10.)/20.),50))
        font = pygame.font.SysFont(None, 18)
        step = font.render("step: "+str(self.timestep), True, (0,0,0))
        self.screen.blit(step, (10,10))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

    def step(self, action):
        l = 0
        r = 0
        if action == 0:
            l = 2.
        elif action == 1:
            r = -1
        else:
            r = 1
        for i in range(15):
            self.agent[0] += cos(radians(self.agent[2]))*l
            self.agent[1] += sin(radians(self.agent[2]))*l
            for j in range(2):
                if self.agent[j] < 10.:
                    self.agent[j] = 10.
                if self.agent[j] > 480.:
                    self.agent[j] = 480.
            self.agent[2] += r
            self.eat()
            self.render()
        self.agent[3] -= 0.01
        self.agent[4] -= 0.01
        state = self.state()
        D = sqrt(self.agent[3]**2 + self.agent[4]**2)
        reward = D - self.last_D
        self.last_D = D
        done = False
        if self.agent[3] < -10. or self.agent[3] > 10. or self.agent[4] < -10. or self.agent[4] > 10:
            done = True
        self.timestep += 1
        return state, reward, done

#env = Environment()
#while True:
#    state, reward, done = env.action(0)
