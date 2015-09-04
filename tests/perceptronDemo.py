from network.perceptron import perceptron
from matplotlib.pylab import *

#predict data label just a joke,and we do not use unittest module to do it.
print 'test 1: predict data label'
p = perceptron(1)
p.init()
p.train([[-0.4, 0.9, 1], [-0.5, 0, 1], [0.6, 0.1, -1]])
print 'predict label of the data is:%d'% p.expectedOutput([-0.2, 0.4])

#classify data by a line
print '\ntest 2: classify data by a line'
trainDataA = [[uniform(-1, 0), uniform(0, 1), 1] for a in xrange(50)]
trainDataB = [[uniform(0, 1), uniform(0, -1), -1] for b in xrange(50)]
print 'train data set:%s'% trainDataB
trainData = trainDataA + trainDataB
p1 = perceptron(1)
p1.init()
p1.train(trainData)
for x in trainData:
    r = p1.expectedOutput(x)
    if r != x[2]:
        print 'error dot:('
    if r == 1:
        plot(x[0], x[1], 'ob')
    else:
        plot(x[0], x[1], 'or')
n = norm(p1.weights) # aka the length of p.w vector
ww = p1.weights / n # a unit vector
ww1 = [ww[1], -ww[0]]
ww2 = [-ww[1], ww[0]]
plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
savefig("../data/test/perceptronClassification_demo.png")
show()