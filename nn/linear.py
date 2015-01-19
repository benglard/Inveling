from nn.utils import shared_normal, shared_zeros
from math import sqrt
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Linear(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        scale = sqrt(1.0/n_in)
        self.W = shared_normal((self.n_in, self.n_out), scale=scale)
        self.b = shared_zeros((self.n_out,))
        self.params = [ self.W, self.b ]
        self.cost = 0

    def forward(self, input):
        return T.dot(input, self.W) + self.b

class Reshape(object):

    def __init__(self, shape):
        self.shape = shape
        self.params = []
        self.cost = 0

    def forward(self, input):
        return T.reshape(input, self.shape)

class Transfer(object):

    def __init__(self, pattern):
        self.pattern = pattern
        self.params = []
        self.cost = 0

    def forward(self, input):
        return input.dimshuffle(self.pattern)

class Dropout(object):

    def __init__(self, p=0.5):
        self.p = p
        self.params = []
        self.cost = 0
        self.training = True
        self.rng = RandomStreams()

    def forward(self, input):
        if self.training:
            return input * self.rng.binomial(
                input.shape,
                p=(1-self.p),
                dtype=theano.config.floatX)
        else:
            return input

class Identity(object):

    def __init__(self):
        self.params = []
        self.cost = 0

    def forward(self, input):
        return input
