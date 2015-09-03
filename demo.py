import perceptron

p = perceptron.singlePerceptron(1)
p.init()
p.train([[-0.4, 0.9, 1], [-0.5, 0, 1], [0.6, 0.1, -1]])
print p.expectedOutput([-0.2, 0.4])
