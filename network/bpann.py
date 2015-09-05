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

    def init(self,inputNodeNum = 4,
        hiddenNodeNum = 5,
        outputNodeNum = 3,
        maxIteration = 1000,
        initWeights_Start = -1,
        initWeights_End = 1,
        learningRate = 0.85
        ):
        self.inputNodeNum = inputNodeNum
        self.hiddenNodeNum = hiddenNodeNum
        self.outputNodeNum = outputNodeNum
        self.maxIteration = maxIteration
        self.learningRate = learningRate

        #inital inputLayer-hiddenLayer and hiddenLayer-outputLayer weights
        self.ihWeights = [[random.uniform(initWeights_Start,initWeights_End) for t in xrange(hiddenNodeNum)]for x in xrange(inputNodeNum)]
        self.hoWeights = [[random.uniform(initWeights_Start,initWeights_End) for t in xrange(outputNodeNum)]for x in xrange(hiddenNodeNum)]

        #to save input and output data of hidden layer and output
        #layer for back propagation
        self.hdata = [0] * self.hiddenNodeNum
        self.odata = [0] * self.outputNodeNum

    def train(self,trainData):
        learned = False
        while not learned:
            for x in trainData:
                self.feedForward(x)
                self.backPropagate(x)
            if self.convergence == 0.0 or \
                self.iteration >= self.maxIteration:
                print 'learned'
                learned = True
            else:
                print 'global error:%d' % self.convergence

    def feedForward(self,trainDataPart):
        for hnode in xrange(self.hiddenNodeNum):
            net = 0
            for x in xrange(len(trainDataPart)-1):
                net += trainDataPart[x] * self.ihWeights[x][hnode]#default bias=0
        self.hdata[hnode] = [net,self.activationFunc(net)]

        for onode in xrange(self.outputNodeNum):
            net = 0
            for x in xrange(len(self.hdata)):
                net += self.hdata[x][1] * self.hoWeights[x][onode]
        self.odata[onode] = [net,self.activationFunc(net)]

    def backPropagate(self,trainDataPart):
        dsigmoid = lambda x:1 - x**2

        #todo:update hiddenLayer-outputLayer weights
        #todo:update inputLayer-hiddenLayer weights
        #cacl global error
        for x in xrange(self.outputNodeNum):
            self.convergence = self.learningRate * (trainDataPart[-1] - self.odata[x][1])**2

    def expectedOutput(self,data):  
        if len(data) != self.inputNodeNum:
            print 'incorrect number of input data'
            #todo

    def activationFunc(self,x):          
        return 1.0/(1.0 + math.exp(-x))