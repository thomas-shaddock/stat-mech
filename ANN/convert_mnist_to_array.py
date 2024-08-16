import numpy as np
import json

with open('mnist_handwritten_train.json') as f:
    mnist_train = json.load(f)

with open('mnist_handwritten_test.json') as f:
    mnist_test = json.load(f)
    

train = np.vstack([np.array(mnist_train[i]['image'],dtype='uint8') for i in range(len(mnist_train))])
test = np.vstack([np.array(mnist_test[i]['image'],dtype='uint8') for i in range(len(mnist_test))])



ls = []
for i in range(len(mnist_train)):
    l = np.zeros(10)
    l[mnist_train[i]['label']] = 1
    ls.append(l)
    
train_l = np.vstack(ls)

ls = []
for i in range(len(mnist_test)):
    l = np.zeros(10)
    l[mnist_test[i]['label']] = 1
    ls.append(l)
    
test_l = np.vstack(ls)

print(train.shape)

np.save('mnist_train',train)
np.save('mnist_train_l',train_l)
np.save('mnist_test',test)
np.save('mnist_test_l',test_l)



