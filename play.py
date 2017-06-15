import gym
import numpy as np
from agent import Agent
from env import Environment
import random
import copy
import csv

env = Environment()
agent = Agent()
epsilon = 0.1
gamma = 0.95
time = 0
last_Q = np.zeros((3),dtype=np.float32)
Q = np.zeros((3), dtype=np.float32)
r = np.zeros((3), dtype=np.float32)
first = True
count = 0
ave = 0.0
data = []
episode = 0

while(True):
    """
    if(count%100==0):
        ave/=100
        if(count!=0):
            with open('data.csv', 'w') as f:
                data.append([time,count,ave])
                writer = csv.writer(f)
                writer.writerows(data)
        print('Epsilon:'+str(epsilon))
        print('Average:'+str(ave))
        ave=0.0
    count+=1
    """

    observation = env.reset()
    for t in range(100000):
        env.render()
        Q = agent.forward(observation)
        if(random.random()>=epsilon):
            action = np.argmax(Q.data)
        else:
            action = random.randint(0,2)
        #epsilon -= 5.0/10**6
        #if(epsilon < 0.1):
        #    epsilon = 0.1

        observation, reward, done = env.step(action)
        if first:
            first=False
        else:
            agent.backward(observation, last_observation, last_action, reward)

        if done:
            print('episode'+str(episode)+':'+str(t))
            episode += 1
            ave += t+1
            time += t+1
            break
        last_observation = observation
        last_action = action
