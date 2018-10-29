import numpy as np
import math
from h1_util import numerical_grad_check




def one_in_k_encoding(vec, k):
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc

def softmax(X):
    res = np.zeros(X.shape)
    res=[np.exp(x)/np.sum(np.exp(x)) for x in X]
    res=np.asarray(res)
    return res

class SoftmaxClassifier():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.W = None
        
    def predict(self, X):
        out = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            out[i]=np.argmax(X[i]@self.W)
        return out
        
    def cost_grad(self, X, y, W):
        cost = np.nan
        grad = np.zeros(W.shape)*np.nan
        Yk = one_in_k_encoding(y, self.num_classes)
        cost=0    
        grad=-((X.T@(Yk-softmax(X@W)))/X.shape[0])
        
        for x, yk in zip(X, Yk):
            lsoft=np.log(softmax(np.dot(x.T, W).reshape((1,W.shape[1]))))
            s = np.dot(lsoft, yk)
            cost +=s
        cost=-cost/len(X)
        
        return cost,grad
    
    def fit(self, X, Y, W=None, lr=0.01, epochs=10, batch_size=16):
        if W is None: W = np.zeros((X.shape[1], self.num_classes))
        history = []
        for i in range(epochs):
            #Shuffle Data
            shuff_X = np.copy(X)
            shuff_Y = np.copy(Y)
            np.random.shuffle(shuff_Y)
            rand = np.random.get_state()
            np.random.shuffle(shuff_X)
            np.random.set_state(rand)
            n = len(shuff_X)
            b = math.ceil(n/batch_size)
            count=0
            for j in range(b):
                final = min(count+batch_size,n)
                batchX = shuff_X[count:final]
                batchY = shuff_Y[count:final]
                cost, grad = self.cost_grad(batchX,batchY,W)
                W+=lr*grad
                history.append(cost)  
                count += batch_size 
        self.W = W
        self.history = history
        
    def score(self, X, Y):
        out = 0
        prob = self.predict(X)
        out = np.mean(prob == Y)
        return out
