import numpy as np
import matplotlib.pyplot as plt


class ANN:

    num_layers = 3
    layer_size = 32
    input_size = 5
    output_size = 10

    network = []

    f = lambda self, x: 1 / (1 + np.exp(x))
    df = lambda self, x: self.f(x) * (1 - self.f(x))

    def __init__(self,num_layers=3,layer_size=32,input_size=5,output_size=10) -> None:
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.generate_layers()


    def generate_layers(self):
        network = []
        a = self.input_size
        b = self.layer_size
        i = 0
        
        while i < self.num_layers:
            if i == self.num_layers - 1:
                b = self.output_size
                
            self.network.append(2*np.random.random([a,b]) - np.ones([a,b]))
            
            a = self.layer_size
            i += 1
        
    
    def evaluate(self, data):
        layers = []
        layers.append(data)
        i = 0
        while i < len(self.network):
            layers.append(self.f(np.matmul(layers[i],self.network[i])))
            i += 1
            
        return layers, layers[-1]
    
    def train(self, training_data, target):
    
        # push forward
        layers, result = self.evaluate(training_data)
        result_error = target.T - result.T # 1 x k
        print('result_error', np.linalg.norm(result_error))
        
        # backpropagate
        errors = [result_error]
        deltas = [result_error * self.df(layers[-1]).T] # (1 x k)

        i = len(self.network) - 2
        while i >= 0:
            #print('a',deltas[0].T.shape,self.network[i+1].T.shape)
            print('deltas[0]')
            errors.insert(0, (deltas[0].T @ self.network[i+1].T)) # (k x 1) (1 x n) so errors[0] is (k x n)
            #print('b',errors[0].shape,self.df(layers[i]).shape)
            deltas.insert(0, (errors[0] * self.df(layers[i+1])).T) # (k x n)
            
            i -= 1
        
        
        i = 0
        while i < len(self.network):
            print('deltas',i,deltas[i].max())
            print('errors',i,errors[i].max())
            print('layers',i,layers[i].max())
            print('network adjustment',i,np.linalg.norm(layers[i].T @ deltas[i].T))
            self.network[i] += layers[i].T @ deltas[i].T
            
            i += 1
        
            
            
            
class td:
    X = [0,0,0,0,0,
        0,0,0,0,1,
        0,0,0,1,0,
        0,0,0,1,1,
        0,0,1,0,0,
        0,0,1,0,1,
        0,0,1,1,0,
        0,0,1,1,1,
        0,1,0,0,0,
        0,1,0,0,1,
        #0,1,0,1,0, #Z, the test, will be this one
        0,1,0,1,1,
        0,1,1,0,0,
        0,1,1,0,1,
        0,1,1,1,0,
        0,1,1,1,1,
        1,0,0,0,0,
        1,0,0,0,1,
        1,0,0,1,0,
        1,0,0,1,1,
        1,0,1,0,0,
        1,0,1,0,1,
        1,0,1,1,0,
        1,0,1,1,1,
        1,1,0,0,0,
        1,1,0,0,1,
        1,1,0,1,0,
        1,1,0,1,1,
        1,1,1,0,0,
        1,1,1,0,1,
        1,1,1,1,0,
        1,1,1,1,1
        ]
    X = np.reshape(X,[31,5])
    Y = np.array([1,0,0,1,0,1,1,0,0,1,#1,
        0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0])
    Z = np.array([0,1,0,1,0])
    Z2 = np.array([1,1,1,1,1])


## Code to train and test neural net code



mnist_train = np.load('mnist_train.npy')
mnist_train_l = np.load('mnist_train_l.npy')
mnist_test = np.load('mnist_test.npy')
mnist_test_l = np.load('mnist_test_l.npy')


nn = ANN(input_size=784,output_size=10)

nn.train(mnist_train,mnist_train_l)
for i in range(10):
    layers, result = nn.evaluate(mnist_test[0])
    #print(np.linalg.norm(mnist_test_l - result))
    print(result)
    
    nn.train(mnist_train,mnist_train_l)

#print(nn.network)