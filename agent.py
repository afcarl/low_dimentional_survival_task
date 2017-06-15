import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import copy

gamma = 0.95

class Agent(chainer.Chain):
    count = 0

    def __init__(self):
        super(Agent, self).__init__(
        l1=L.Linear(2+19*3, 2000),
        l2=L.Linear(2000, 3)
        )

    def forward(self,observation):
        x = np.zeros((1,2+19*3),dtype=np.float32)
        x[0] = observation
        x = Variable(x)
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y

    def backward(self, ob ,last_ob,action,r):
        q_dash = self.forward(last_ob)
        q = self.forward(ob)
        target = np.array(copy.deepcopy(q.data),dtype=np.float32)
        target[0][action] = r + gamma * np.amax(q_dash.data)
        td = Variable(target) - q# TD error
        td_tmp =  td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
        zero_val = Variable(np.zeros((1, 3), dtype=np.float32))
        model.cleargrads()
        loss = chainer.functions.mean_squared_error(td_clip, zero_val)
        loss.backward()
        optimizer.update()
        #self.count+=1
        #if(self.count%1000==0):
        #    serializers.save_hdf5('model.model', model)
        #    serializers.save_hdf5('state.state', optimizer)

model = Agent()
#optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
optimizer = optimizers.Adam()
optimizer.setup(model)
print('loading model')
#serializers.load_hdf5('model.model', model)
#serializers.load_hdf5('state.state', optimizer)
print('loaded')
