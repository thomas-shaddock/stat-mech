import numpy as np
import matplotlib.pyplot as plt


class ANN:

    num_layers = 3
    layer_size = 32
    input_size = 5
    output_size = 10
    learning_rate = 0.05

    network = []

    f = lambda self, x: 1 / (1 + np.exp(-x))
    df = lambda self, x: self.f(x) * (1 - self.f(x))
    # df = lambda self, x: x * (1 - x)

    def __init__(self,num_layers=3,layer_size=32,input_size=5,output_size=1) -> None:
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
                
            self.network.append(2*np.random.random([b,a]) - np.ones([b,a]))
            
            a = self.layer_size
            i += 1
        
    
    def evaluate(self, data):
        activation = []
        activation.append(data)
        i = 0
        while i < len(self.network):
            # print(self.network[i].shape, activation[i].shape)
            activation.append(self.f(self.network[i] @ activation[i]))
            i += 1
            
        return activation, activation[-1]
    
    def train(self, training_data, target):
    
        # k = size of training dataset
        # n_l = size of layer l
        # note that network[l] has size (n_l+1,n_l)
        
        
        # push forward
        activation, result = self.evaluate(training_data) # (n,k)
        result_error = -target + result # (n_L,k)
        # print('result_error', np.max(result_error))
        
        # backpropagate
        
        
        # errors = [result_error]
        # deltas = [result_error * self.df(layers[-1]).T] # (1,k)
        deltas = [self.df(result_error)] # (n_L,k)
        grads = [deltas[0] @ activation[-2].T] # (n_L,k) (k,n_L) = (n_L,n_L-1)
        
        i = len(self.network) - 2
        while i >= 0:
            # print(self.network[i+1].T.shape,deltas[0].shape)
            deltas.insert(0, self.df(self.network[i+1].T @ deltas[0]))
            # print(deltas[0].shape, activation[i].T.shape)
            grads.insert(0, deltas[0] @ activation[i].T)
            
            # errors.insert(0, (deltas[0].T @ self.network[i+1].T)) # (k x 1) (1 x n) so errors[0] is (k x n)
            # deltas.insert(0, (errors[0] * self.df(layers[i+1])).T) # (k x n)
            
            i -= 1
            
        # print('errors',[np.max(e) for e in errors])
        # print('deltas',[np.max(d) for d in deltas])
        
        
        i = 0
        while i < len(self.network):
            #print('deltas',i,deltas[i].max())
            #print('errors',i,errors[i].max())
            #print('layers',i,layers[i].max())
            # print('max network adjustment layer',i,np.max(self.learning_rate * grads[i]))
            print(nn.network)
            self.network[i] -= self.learning_rate * grads[i]
            
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
    X = np.reshape(X,[31,5]).T
    Y = np.array([1,0,0,1,0,1,1,0,0,1,#1,
        0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]).reshape(1,-1)
    Z = np.array([0,1,0,1,0]).T
    Z2 = np.array([1,1,1,1,1])

class xor:
    X = np.array([[0,0],[0,1],[1,0],[1,1]]).T
    Y = np.array([0,1,1,0])
    Z = np.array([1,1]).T

## Code to train and test neural net code



mnist_train = np.load('mnist_train.npy').astype('float64')/255
mnist_train_l = np.load('mnist_train_l.npy')
mnist_test = np.load('mnist_test.npy').astype('float64')/255
mnist_test_l = np.load('mnist_test_l.npy')


#nn = ANN(input_size=784,output_size=10)

# nn.train(mnist_train,mnist_train_l)
# for i in range(10):
#     layers, result = nn.evaluate(mnist_test[0])
#     #print(np.linalg.norm(mnist_test_l - result))
#     print(result)
    
#     nn.train(mnist_train,mnist_train_l)

#print(nn.network)

# layers, result = nn.evaluate(mnist_test[0])
# #print(np.linalg.norm(mnist_test_l - result))
# print(result,mnist_test_l[0])

nn = ANN(num_layers=2,layer_size=4,input_size=2,output_size=1)

Z2 = np.array([1,0,1,1,0])

for i in range(50):
    print(nn.evaluate(xor.Z)[1])
    nn.train(xor.X,xor.Y)