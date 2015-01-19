import theano.tensor as T

class NonLinear(object):

    def __init__(self, f):
        self.f = f
        self.params = []
        self.cost = 0

    def forward(self, input):
        return self.f(input)

Tanh = lambda: NonLinear(T.tanh)
Sigmoid = lambda: NonLinear(T.nnet.sigmoid)
ReLU = lambda: NonLinear(lambda x: ((x + abs(x)) / 2.0))
Softmax = lambda: NonLinear(T.nnet.softmax)
