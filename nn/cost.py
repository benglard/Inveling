import theano.tensor as T

class CostLayer(object):

    def __init__(self, f):
        self.f = f
        self.params = []

    def forward(self, input, target):
        self.input = input
        return self.f(input, target)

MSE = lambda: CostLayer(lambda x, t: T.mean((x - t) ** 2))