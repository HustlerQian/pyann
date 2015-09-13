# ![logo](http://i1.tietuku.com/876fae1c923a713f.png)
![license](https://img.shields.io/github/license/mashape/apistatus.svg)![pythontype](https://img.shields.io/badge/python-2.7-blue.svg)![status](https://img.shields.io/badge/status-updating-orange.svg)<br>
pyann is a simple but stronger artificial neural network library for python. I refer to KISS(Keep it simple,stupid) Philosophy as the core principle of it, so you can read or modify source code easily. For user,it's easy to build many different neural networks by a little code:
```python
#create a perceptron model and train it through train data
import network.perceptron as ptron

trainData = [[..,..,label],[..,..,label]]

net = ptron.perceptron()
net.init()
net.train(trainData)
net.expected([..,..]) #will print predicted class
```
Now this project is still updating, also I hope you can help me complete this library and make it perfect!<br>
For Chinese,see [READMECN.md](https://github.com/racaljk/pyann/blob/master/READMECN.md).

## Artificial Neural Network Models
* **perceptron model**<br>
(`core code:network/perceptron.py` `demo:/tests/perceptronDemo.py`)
![perceptron](https://raw.githubusercontent.com/racaljk/pyann/master/data/test/perceptronClassification_demo.png)
* **bpnn model** <br>
waiting for completing

## Depencies
* python2.7
* numpy https://github.com/numpy/numpy [optional for demo]
* matplotlib http://matplotlib.org/ [optional for demo]

## Contact Me
1. [Tieba](http://tieba.baidu.com/home/main?un=%CF%C0%B5%C1%D0%A1%B7%C9%BB%FA)
2. [twitter](http://twitter.com/cthulhujk)
3. [e-mail](mailto:1948638989@qq.com)

## License
pyann is under The MIT License (MIT),you can see [MIT License](https://github.com/racaljk/pyann/blob/master/LICENSE) file for details,