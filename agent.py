import numpy as np
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"
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
        self.count = 0
        self.loss = 0.
        super(Agent, self).__init__(
        l1=L.Linear(4+19*3, 2000),
        l2=L.Linear(2000, 3),
        b1=L.BatchNormalization(2000)
        )

    def forward(self,observation):
        x = np.zeros((1,4+19*3),dtype=np.float32)
        x[0] = observation
        x = Variable(x)
        #h1 = F.relu(self.b1(self.l1(x)))
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y

    def backward(self, ob_dash ,ob,action,r):
        self.count+=1
        q_dash = self.forward(ob_dash)
        q = self.forward(ob)
        target = np.array(copy.deepcopy(q.data),dtype=np.float32)#q
        target[0][action] = r + gamma * np.amax(q_dash.data)#q_dash.data)
        if self.count % 1000 == 0:
            print('q:',q.data[0][action],'target:',target[0][action],'reward:',r,'ave_loss',self.loss/1000.)
            self.loss = 0.
        td = Variable(target) - q#q TD error
        self.loss += target[0][action] - q.data[0][action]
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
cuda.get_device(0).use()
model.to_gpu()
#optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
optimizer = optimizers.Adam(alpha=0.01, eps=1.0)#Adam()
optimizer.setup(model)
print('loading model')
#serializers.load_hdf5('model.model', model)
#serializers.load_hdf5('state.state', optimizer)
print('loaded')
