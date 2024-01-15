# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:18:43 2022

@author: turinici
"""
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#
# Load the iris dataset
#
iris = datasets.load_iris()
X = iris.data
n_iris = X.shape[0]
y = iris.target
# tranforms y in vector of len=3 with 0/1 : hot encoding
nb_classes = len(np.unique(y))
one_hot_targets = np.eye(nb_classes)[y]  # matrice de taille  (150, 3)


def softmax(x):
    x = x-np.max(x)
    return np.exp(x)/np.sum(np.exp(x))


# function to create a dense layer
def create_dense_layer(n_input, n_output):
    # matrice n_output,n_input) de v.a. normales de moyenne 0 et variance "1/n_input"
    weights = np.random.randn(n_output, n_input)/np.sqrt(n_input)
    # vecteur colonne nul de n_output composantes
    biases = np.zeros((n_output, 1))
    return weights, biases


def ReLU(input_array):
    return np.maximum(0, input_array)


dim1, dim2, dim3, dim4 = 4, 5, 7, 3
weights1, biases1 = create_dense_layer(dim1, dim2)
weights2, biases2 = create_dense_layer(dim2, dim3)
weights3, biases3 = create_dense_layer(dim3, dim4)
losses = []

alpha = 0.001  # put alpha=0 to compare with no training
# train loop
n_iter = 20000
for iter in range(n_iter):
    # alpha=10/n_iter# this decay schedule seems ok
    sample_index = np.random.choice(n_iris)
    Y0 = X[[sample_index], :].T
    # FORWARD STEP
    # dense layer 1
    Y1tilde = weights1@Y0+biases1
    # activation 1: ReLU
    Y1 = ReLU(Y1tilde)
    # dense layer 2
    Y2tilde = weights2@Y1+biases2
    # activation 2: ReLU
    Y2 = ReLU(Y2tilde)
    # dense layer 3
    Y3tilde = weights3@Y2+biases3

    # final computations of the fwd step
    label = one_hot_targets[[sample_index], :].T
    q = softmax(Y3tilde)
    loss_val = -np.sum(label*np.log(q))
    losses.append(loss_val)

    # utiliser formule
    dY3tilde = -label+q

    # gradient on parameters layer 3
    dweights3 = dY3tilde@Y2.T
    dbiases3 = dY3tilde
    # gradient on values activation 2
    dY2 = weights3.T@dY3tilde
    dY2tilde = dY2.copy()
    dY2tilde[Y2tilde < 0] = 0

    # gradient on parameters layer 2
    dweights2 = dY2tilde@Y1.T
    dbiases2 = dY2tilde
    # gradient on values activation 1
    dY1 = weights2.T@dY2tilde
    dY1tilde = dY1.copy()
    dY1tilde[Y1tilde <= 0] = 0

    # gradient on parameters layer 1
    dweights1 = dY1tilde@Y0.T
    dbiases1 = dY1tilde

    # update weights and biases
    weights1 -= alpha*dweights1
    biases1 -= alpha*dbiases1
    weights2 -= alpha*dweights2
    biases2 -= alpha*dbiases2
    weights3 -= alpha*dweights3
    biases3 -= alpha*dbiases3

# plt.loglog(losses)
# plt.plot(losses)
# plt.title('losses')
# plt.pause(1)

# evaluate accuracy
qresults = []
count = 0
for nn in range(n_iris):
    # propagate example ii

    Y0 = X[[nn], :].T
    # dense layer 1
    Y1tilde = weights1@Y0+biases1
    # activation 1: ReLU
    Y1 = ReLU(Y1tilde)
    Y2tilde = weights2@Y1+biases2  # dense layer 2
    Y2 = ReLU(Y2tilde)  # activation 2: ReLU
    Y3tilde = weights3@Y2+biases3  # dense layer 3
    q = softmax(Y3tilde)
    qresults.append(q.copy())
    if np.all([q[y[nn], 0] > p for jj, p in enumerate(q[:, 0]) if jj != y[nn]]):
        count = count+1
print('accuracy =', np.round(100*count/n_iris, 2), "\b%")


fig, ax = plt.subplots(1, 2)
ax[0].plot(losses)
ax[0].set_title('losses')
ax[1].hist(np.argmax(qresults, axis=1))
ax[1].hist(y, alpha=0.5)
ax[1].legend(['predictions', 'true values'])
ax[1].set_title('predictions')
plt.show()
