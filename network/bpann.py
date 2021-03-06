"""
The MIT License (MIT)
Copyright (c) 2015 Abstract Operator
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
__author__=['racaljk(Abatract Operator);1948638989@qq.com']

import random
import math

class bpann(object):
    """back propagation artificial neural network"""

    def __init__(self,debugMode=False):
        """
        :param debugMode: if printing debug info
        """
        self.debug = debugMode
        self.iteration = 0
        self.convergence =0.0

    def init(self,inputLayerNodeNum = 4,
        hiddenLayerNodeNum = 5,
        outputLayerNodeNum = 3,
        maxIteration = 1000,
        biasRange = [0,1],
        weightRange = [-1,1]
        learningRate = 0.85
        ):
        #constant
        self.inode = inputLayerNodeNum
        self.hnode = hiddenLayerNodeNum
        self.onode = outputLayerNodeNum
        self.maxIter = maxIteration
        self.lr = learningRate
        
        #initial bias
        self.hb = random.uniform(biasRange[0],biasRange[1])
        self.ob = random.uniform(biasRange[0],biasRange[1])

        #initial inputLayer-hiddenLayer and hiddenLayer-outputLayer weights
        self._init_hWeights()
        self._init_oWeights()

        #save input and output data 
        self.hdata = [0] * self.hnode
        self.odata = [0] * self.onode

    def train(self,trainData):
        learned = False
        while not learned:
            for record in trainData:
                self.feedForward(record)
                self.backPropagate(record)
            
    def feedForward(self,data):
        for x in xrange(self.hnode):
            s = 0
            for t in xrange(len(data)-1):
                s += data[t] * self.hw[t][x] + self.hb
            self.hdata[x] = [s,self._sigmoid(s)]

        for x in xrange(self.onode):
            s = 0
            for t in xrange(self.hnode):
                s += self.hdata[t][1] * self.ow[t][x] +self.ob
            self.odata[x] = [s,_sigmoid(s)]

    def backPropagate(self,data):
        Eo = [0] * self.onode
        Eh = [0] * self.hnode
        #update hiddenLayer-outputLayer weights
        for x in xrange(self.onode):
            Eo[x] = self.odata[x][1] * (1-self.odata[x][1]) * (data[-1] - self.odata[x][-1])
            for t in xrange(self.hnode):
                self.ow[t][x] += self.lr * self.hdata[t][1] * Eo[x]
                self.ob += self.lr * Eo[x]  
        #update inputLayer-hiddenLayer weights
        for x in xrange(self.hnode):
            s =0
            for p in xrange(self.onode):
                s += self.ow[x][p] * Eo[p]
            Eh[x] = self.hdata[x][1] * (1-self.hdata[x][1]) * s
            for t in xrange(self.inode):
                self.hw[t][x] += self.lr * data[t] * self.Eh[x] 
                self.hb +=self.lr * Eh[x]

        #todo:cacl global error

    def expectedOutput(self,data):
        pass

    def _sigmoid(self,x):
        return 1.0/(1.0 + math.exp(-x))
    def _init_hWeights():
        self.hw = [[random.uniform(weightRange[0],weightRange[1]) \
        for t in xrange(self.hnode)]for x in xrange(self.inode)]
    def _init_oWeights():
        self.ow = [[random.uniform(weightRange[0],weightRange[1]) \
        for t in xrange(self.onode)]for x in xrange(self.hnode)] 

