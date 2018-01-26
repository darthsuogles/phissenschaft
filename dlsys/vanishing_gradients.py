"""
Inspecting vanishing gradients for simple neural net
"""

# Setup
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import namedtuple

sns.set(rc={"figure.figsize": (8.0, 6.0)})
sns.set(style="darkgrid", color_codes=True)

# xs = sorted(np.random.randn(100))
# ys = np.sinc(xs)
# plt.plot(xs, ys); plt.show()

class SpiralDataset(object):
    def __init__(self, N, D, K, seed=2702):
        np.random.seed(seed)
        self.N = N  # samples per class
        self.D = D  # ambient space dimension
        self.K = K  # number of classes
        self._gen_data()

    def _gen_data(self):
        self.X = np.zeros((self.N * self.K, self.D))
        self.num_train_examples = self.X.shape[0]
        self.y = np.zeros(self.N * self.K, dtype=np.uint8)

        for class_label in range(self.K):
            ix = range(self.N * class_label, self.N * (class_label + 1))
            r = np.linspace(0., 1., self.N)
            t = np.linspace(class_label * 4, (class_label + 1) * 4, self.N) + \
                np.random.randn(self.N) * 0.2
            self.X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            self.y[ix] = class_label

    def show(self, ax):
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y,
                    s=40, cmap=plt.cm.Spectral)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

# # When having a bunch of plots in a grid (via subplots or gridspec),
# # we can plot the title as the ylabel of the left-most plot.
# # http://matplotlib.org/users/text_props.html
# ax.set_ylabel('<text_of_the_title>', rotation = 'horizontal', horizontalalignment = 'right');

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_grad(x):
    return x * (1. - x)

def relu(x):
    return np.maximum(0., x)


class ThreeLayerNet(object):

    def __init__(self, W1, W2, W3, b1, b2, b3, reg, NONLINEARITY):
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.reg = reg
        self.NONLINEARITY = NONLINEARITY
        self.activation = relu if self.NONLINEARITY == 'RELU' else sigmoid

    def forward(self, dataset):
        # Forward pass
        self.fc1 = self.activation(np.dot(dataset.X, self.W1) + self.b1)
        self.fc2 = self.activation(np.dot(self.fc1, self.W2) + self.b2)
        self.scores = np.dot(self.fc2, self.W3) + self.b3
        exp_scores = np.exp(self.scores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, dataset, iteration, step_size):
        num_examples = dataset.X.shape[0]
        ix = range(num_examples)

        # Compute loss
        correct_logprobs = -np.log(self.probs[ix, dataset.y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = .5 * self.reg * np.sum(self.W1 * self.W1) + \
                   .5 * self.reg * np.sum(self.W2 * self.W2) + \
                   .5 * self.reg * np.sum(self.W3 * self.W3)
        self.tot_loss = data_loss + reg_loss

        if 0 == iteration % 2000:
            print("Iteration {:d}, loss {:.6f}".format(iteration, self.tot_loss))

        # Backward pass
        # 1. Gradient: cross-entropy loss for soft-max
        d_scores = self.probs
        d_scores[ix, dataset.y] -= 1
        d_scores /= num_examples

        # 2. Gradient: exposure layer
        d_W3 = np.dot(self.fc2.T, d_scores)
        d_b3 = np.sum(d_scores, axis=0, keepdims=True)

        if 'RELU' == self.NONLINEARITY:
            d_fc2 = np.dot(d_scores, self.W3.T)
            d_fc2[self.fc2 <= 0] = 0
            d_W2 = np.dot(self.fc1.T, d_fc2)
            d_b2 = np.sum(d_fc2, axis=0)
            d_fc1 = np.dot(d_fc2, self.W2.T)
            d_fc1[self.fc1 <= 0] = 0

        elif 'SIGM' == self.NONLINEARITY:
            d_fc2 = d_scores.dot(self.W3.T) * sigmoid_grad(self.fc2)
            d_W2 = np.dot(self.fc1.T, d_fc2)
            d_b2 = np.sum(d_fc2, axis=0)
            d_fc1 = d_fc2.dot(self.W2.T) * sigmoid_grad(self.fc1)

        d_W1 = np.dot(dataset.X.T, d_fc1)
        d_b1 = np.sum(d_fc1, axis=0)

        # Add regulariation
        d_W3 += self.reg * self.W3
        d_W2 += self.reg * self.W2
        d_W1 += self.reg * self.W1

        self.grads = {'W1': d_W1, 'W2': d_W2, 'W3': d_W3,
                      'b1': d_b1, 'b2': d_b2, 'b3': d_b3}

        self.W1 += -step_size * d_W1
        self.b1 += -step_size * d_b2
        self.W2 += -step_size * d_W2
        self.b2 += -step_size * d_b2
        self.W3 += -step_size * d_W3
        self.b3 += -step_size * d_b3

        return self.grads


N = 100
D = 2
K = 3
dataset = SpiralDataset(N, D, K)
# fig, ax = plt.subplots()
# dataset.show(ax)

h_fc1 = 50
h_fc2 = 50

model = {}
model['W1'] = 0.1 * np.random.randn(D, h_fc1)
model['b1'] = np.zeros((1, h_fc1))
model['W2'] = 0.1 * np.random.randn(h_fc1, h_fc2)
model['b2'] = np.zeros((1, h_fc2))
model['W3'] = 0.1 * np.random.randn(h_fc2, K)
model['b3'] = np.zeros((1, K))
model['reg'] = 1e-4
model['NONLINEARITY'] = 'SIGM'

nnet = ThreeLayerNet(**model)

for i in range(50000):
    nnet.forward(dataset)
    nnet.backward(dataset, i, 0.1)

# Summarizing
probs = nnet.forward(dataset)
preds = np.argmax(probs, axis=1)
print('Prediction accuracy {:.4f}'.format(
    np.sum(dataset.y == preds) / preds.shape[0]))

# Plot the classification contour
x_min, x_max = dataset.X[:, 0].min() - 1, dataset.X[:, 0].max() + 1
y_min, y_max = dataset.X[:, 1].min() - 1, dataset.X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

grid_data = namedtuple('Dataset', ['X'])(X=np.c_[xx.ravel(), yy.ravel()])
Z = nnet.forward(grid_data)
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y,
            s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
