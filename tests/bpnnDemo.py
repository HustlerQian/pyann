from network.bpnn import bpnn

print 'test 1:'
trainData = []
with open('../data/test/iris.data') as data:
    line = data.readlines()
    for x in xrange(0,50):
        line[x] = line[x].replace('Iris-setosa','1')
        line[x] = line[x].replace('\n','')
        recordList = line[x].split(',')
        for y in xrange(len(recordList)):
            recordList[y] = float(recordList[y])
        trainData.append(recordList)
    for x in xrange(50,100):
        line[x] =line[x].replace('Iris-versicolor','2')
        line[x] = line[x].replace('\n','')
        recordList = line[x].split(',')
        for y in xrange(len(recordList)):
            recordList[y] = float(recordList[y])
        trainData.append(recordList)
    for x in xrange(100,150):
        line[x] =line[x].replace('Iris-virginica','3')
        line[x] = line[x].replace('\n','')
        recordList = line[x].split(',')
        for y in xrange(len(recordList)):
            recordList[y] = float(recordList[y])
        trainData.append(recordList)

bp =bpnn()
bp.init()
bp.train(trainData)

