'''
Created on Oct 30, 2013
@author: Hrishikesh
'''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

def _binary_KL_divergence(p, p_hat):
    return (p * np.log(p / p_hat)) + ((1 - p) * np.log((1 - p) / (1 - p_hat)))

def _sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))
    
def _der_sigmoid(X):
    return X * (1 - X)

class Autoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_hidden=25, learning_rate=0.3, alpha=3e-3, beta=3, sparsity=0.1, max_iter=20, verbose=False, random_state=None):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.beta = beta
        self.sparsity = sparsity
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        
    def _init_weights(self, n_features):
        """
        Weights' initilization with random values
        """
        rng = check_random_state(self.random_state)
        self.weight_hidden_ = rng.uniform(-0.5, 0.5, (n_features, self.n_hidden))  # input to hidden
        self.weight_output_ = rng.uniform(-0.5, 0.5, (self.n_hidden, n_features))  # hidden to output
        # randomly initialized bias vectors
        self.bias_hidden_ = rng.uniform(-0, 0, self.n_hidden)  # input to hidden bias vector
        self.bias_output_ = rng.uniform(-0, 0, n_features)  # hidden to output bas vector
    
    def transform(self, X):
        # activation
        return _sigmoid(np.dot(X, self.weight_hidden_) + self.bias_hidden_)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, y=None):
        '''n_f, n_s = X.shape
        n_features = n_s
        n_samples = n_f'''
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        self._backprop_lbfgs(
                X, n_features, n_samples)
        return self
    
    def _unpack(self, theta, n_features):
        N = self.n_hidden * n_features
        self.weight_hidden_ = np.reshape(theta[:N],
                                      (n_features, self.n_hidden))
        self.weight_output_ = np.reshape(theta[N:2 * N],
                                      (self.n_hidden, n_features))
        self.bias_hidden_ = theta[2 * N:2 * N + self.n_hidden]
        self.bias_output_ = theta[2 * N + self.n_hidden:]

    def _pack(self, W1, W2, b1, b2):
        return np.hstack((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))
    
    def _cost_grad(self, theta, X, n_features, n_samples):
        self._unpack(theta, n_features)
        cost, W1grad, W2grad, b1grad, b2grad = self.backprop(X, n_features, n_samples)
        return cost, self._pack(W1grad, W2grad, b1grad, b2grad)
    
    def backprop(self, X, n_features, n_samples):
        # Forward pass
        activation_hidden = _sigmoid(np.dot(X, self.weight_hidden_)
                                      + self.bias_hidden_)
        activation_output = _sigmoid(np.dot(activation_hidden, self.weight_output_)
                                      + self.bias_output_)
        #print activation_output.shape
        # Compute average activation of hidden neurons
        p = self.sparsity
        p_hat = np.mean(activation_hidden, 0)
        p_delta = self.beta * ((1 - p) / (1 - p_hat) - p / p_hat)
        # Compute cost 
        diff = X - activation_output
        cost = np.sum(diff ** 2) / (2 * n_samples)
        # Add regularization term to cost 
        cost += (0.5 * self.alpha) * (np.sum(self.weight_hidden_ ** 2) + np.sum(self.weight_output_ ** 2))
        # Add sparsity term to the cost
        cost += self.beta * np.sum(_binary_KL_divergence(p, p_hat))
        # Compute the error terms (delta)
        delta_output = -diff * _der_sigmoid(activation_output)
        delta_hidden = ((np.dot(delta_output, self.weight_output_.T) + p_delta)) * _der_sigmoid(activation_hidden)
        # Get gradients
        W1grad = np.dot(X.T, delta_hidden) / n_samples 
        W2grad = np.dot(activation_hidden.T, delta_output) / n_samples
        b1grad = np.mean(delta_hidden, 0) 
        b2grad = np.mean(delta_output, 0) 
        # Add regularization term to weight gradients 
        W1grad += self.alpha * self.weight_hidden_
        W2grad += self.alpha * self.weight_output_
        return cost, W1grad, W2grad, b1grad, b2grad

    def _backprop_lbfgs(self, X, n_features, n_samples):
        # Pack the initial weights 
        # into a vector
        initial_theta = self._pack(
            self.weight_hidden_,
            self.weight_output_,
            self.bias_hidden_,
            self.bias_output_)
        # Optimize the weights using l-bfgs
        optTheta, cost, _ = fmin_l_bfgs_b(
            func=self._cost_grad,
            x0=initial_theta,
            maxfun=self.max_iter,
            disp=self.verbose,
            args=(X,
                n_features,
                n_samples))
        # Unpack the weights into their
        # relevant variables
        self._unpack(optTheta, n_features)
        #print 'theta', optTheta
        print 'cost', cost
    
    def _predict(self, data):
        X = data
        samples, features = X.shape
        #n_samples, n_features = X.shape
        weight_output = self.weight_output_.T
        self.weight_output_ = weight_output[:features].T
        activation_hidden = _sigmoid(np.dot(X, self.weight_hidden_[:features])
                                      + self.bias_hidden_)
        activation_output = _sigmoid(np.dot(activation_hidden, self.weight_output_[:])
                                      + self.bias_output_[:features])
        #activation_output = np.reshape(activation_output, newshape, order)
        print activation_output.shape
        #activation_output = np.max(activation_output, axis=0)
        #X = np.max(X, axis=0)
        #print activation_output.shape
        print X.shape
        X = np.array(X)
        activation_output = np.array(activation_output)
        X = X.reshape([samples*features,1])
        #match = (activation_output[:] == X[:]).astype(int)
        #print match.shape
        #acc = np.sum(match[:])/features
        for j in range(0,samples):
            for i in range(0, features):
                activation_output[j][i] = 1 if activation_output[j][i]>0.5 else 0
        activation_output = activation_output.reshape([samples*features,1])
        #match = np.empty([samples*features,1])
        s= 0
        for i in range(0, samples*features):
            s = s + np.power(activation_output[i][0] - X[i][0],2)
            #print s, activation_output[i][0], X[i][0]
            #match[i] = (activation_output[i] == X[i]).astype(int)
        print 's', samples*features
        print 'sa', s
        s= s/(samples*features)
        print 'MSE:\t', s
        print 'RMSE:\t', np.sqrt(s)
'''if __name__ == '__main__':
    #from sklearn.linear_model import SGDClassifier
    #from sklearn.datasets import fetch_mldata
    #import random
    # Download dataset
    print "fetch"
    #mnist = fetch_mldata('MNIST original')
    print "fetched"
    import time
    s = time.time()
    
    X, y = mnist.data, mnist.target
    print X.shape
    print y.shape
    random.seed(1)
    indices = np.array(random.sample(range(70000), 1000))
    X, y = X[indices].astype('float64'), y[indices]
    print X.shape
    # Scale values in the range [0, 1]
    X = X / 255
    import cPickle as pickle
    X = pickle.load( open( "train_sequence_matrix.pkl", "rb" ) )
    print X.shape
    ae = Autoencoder(max_iter=200, sparsity=0.001,
                                    beta=0.3, n_hidden=80, alpha=3e-3,
                                    verbose=False, random_state=1)
    # Train and extract features
    extracted_features = ae.fit_transform(X)
    print extracted_features.shape
    print time.time() - s, 'seconds'
    defcompute(tName, h):
        print tName
        ae = Autoencoder(max_iter=200, sparsity=0.1,
                                    beta=0.3, n_hidden=h, alpha=3e-3,
                                    verbose=False, random_state=1)
        # Train and extract features
        extracted_features = ae.fit_transform(X)
        print extracted_features.shape
        print time.time() - s
    import thread
    try:
        for i in range(5,1,35):
            print i;
            thread.start_new_thread( compute, ("Thread-1", i))
            thread.start_new_thread( compute, ("Thread-2", 80-i))
#//        thread.start_new_thread( compute, ("Thread-3", 35))
    except:
        print "Error: unable to start thread"
    while 1:
        pass
    
    clf = SGDClassifier(random_state=3)
    clf.fit(X, y)
    print 'SGD on raw pixels score: ', \
              clf.score(X, y)
    clf.fit(extracted_features, y)
    print 'SGD on extracted features score: ', \
              clf.score(extracted_features, y)'''