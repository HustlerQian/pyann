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

class perceptron(object):
    def __init__(self,debugMode=False):
        """
        :param debugMode: if printing debug info
        """
        self.debug = debugMode
        self.iteration = 0

    def init(self,learningRate=0.1,maxIteration = 1000,initWeights_Start=0,initWeights_End=1):
        """These paramaters determine final effect.
        :param learningRate:learning rate,Usually it is between 0 and 1.
        :param maxIteration: if this model can not fitting data well,the
        maxIteration tells  when will it end.
        :param initWeights_Start:take initWeights_Start as the start of the random value range
        :param initWeights_End:take initWeights_End as the end of the random value range
        :rtype : void
        """
        self.theta = learningRate
        self.maxIteration = maxIteration
        self.weights = [random.uniform(initWeights_Start,initWeights_End) for _ in xrange(2)]

    def train(self,trainData):
        """
        :param trainData:
            The training sample data's structure is x=[[-0.49, 0.17, 1], [-0.07, 0.58, 1]]
            it means [[feature_x,feature_y,label]]
        :rtype: void;
        """
        learned = False
        while not learned:
            convergence = 0.0
            for x in trainData:
                ret = self.expectedOutput(x)
                if x[2] != ret:
                    err = x[2] - ret
                    self._updateWeight(err,x)
                    convergence += err**2 #convergence += abs(err)
            self.iteration += 1
            if convergence == 0.0 or \
                self.iteration >= self.maxIteration:
                if self.debug:
                    print 'iterations:%s'% self.iteration
                    print 'weight1:%s\nweight2:%s'% (self.weights[0],self.weights[1])
                    if round(self.weights[0],2) >= 0:
                        print 'linear model: y='+str(round(self.weights[1],2))+'x+'+str(round(self.weights[0],2))
                    else:
                        print 'linear model: y='+str(round(self.weights[1],2))+'x'+str(round(self.weights[0],2))
                learned = True

    def expectedOutput(self,data):
        """Print expected output.
        :param data:like x=[feature_x,feature_y];but when it was called
        by perceptron.train function,its structure is x=[feature_x,feature_y,label]
        :return:label
        :rtype: int
        """
        sum = 0
        for x in xrange(2):
            sum += data[x] * self.weights[x]
        if sum >=0:
            return 1
        else:
            return -1

    def _updateWeight(self,err,trainDataPart):
        for x in xrange(2):
            self.weights[x] += self.theta * err * trainDataPart[x]