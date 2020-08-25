import numpy as np

np.random.seed(0)

def sigmoid(x):
    out = 1 / (1+np.exp(-x))
    return out

def relu(x):
    out = x
    out[out<-1] = -1
    out[out> -1 and out < 0] = 0
    return out

def tanh(x):
    out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return out

class Network(object):
    def __init__(self, shape, weights_path=None):
        self.shape = shape
        self.weights = []
        self.bias = []
        
        for i in range(len(self.shape)-1):
            self.weights.append(np.random.uniform(-1, 1, (self.shape[i], self.shape[i+1])))
            
        for i in range(len(self.shape)-1):
            self.bias.append(np.random.uniform(-1, 1, self.shape[i+1]))

        if weights_path:
            temp = np.genfromtxt(weights_path, delimiter=" ")
            self._import(temp)

        self.layers = []
        for i in range(len(self.shape)):
            self.layers.append(np.zeros(self.shape[i]))

    def forward_propagation(self, inputs):
        self.layers[0] = inputs
        for i in range(len(self.shape)-1):
            self.layers[i+1] = np.dot(self.layers[i], self.weights[i])
            self.layers[i+1] = self.layers[i+1] + (self.bias[i])
            self.layers[i+1] = sigmoid(self.layers[i+1])
        return self.layers[len(self.layers)-1]

    def export(self):
        vect = []
        for layer in self.weights:
            for k in layer:
                for j in k: 
                    vect.append(j)
        for i in self.bias:
            for k in i:
                vect.append(k) 

        return np.array(vect)

    def _import(self, vect):
        s = 0
        for layer in range(len(self.shape)-1):
            for k in range(len(self.weights[layer])):
                for j in range(len(self.weights[layer][k])):    
                    self.weights[layer][k][j] = vect[s]
                    s += 1
        for i in range(len(self.shape)-1):
            for k in range(len(self.bias[i])):
                self.bias[i][k] = vect[s] 
                s += 1  