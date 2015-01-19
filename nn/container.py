import theano, numpy, cPickle
import theano.tensor as T
from nn.utils import shared_zeros

theano.config.exception_verbosity='high'
theano.config.compute_test_value='raise'

class Container(object):

    def __init__(self):
        self.input = T.matrix()
        self.input.tag.test_value = numpy.random.randn(14, 4708).astype(theano.config.floatX)
        self.output = self.input.copy()
        self.target = T.matrix()
        self.target.tag.test_value = numpy.random.randn(300, 100).astype(theano.config.floatX)
        self.layers = []
        self.cost = 0.0

        self.momentum = 0.9
        self.clipping = 15
        self.lr = 0.1
        self.l2_decay = 0.001

    def add(self, layer, cost=False):
        if cost:
            self.cost = layer.forward(self.output, self.target)
            print 'Compiling'
            self.params()
            self.grads()
            self.updates()
        else:
            self.output = layer.forward(self.output)
            print self.output.tag.test_value.shape#, self.output.tag.test_value
            self.layers.append(layer)
        
    def params(self):
        self.ps = sum([ l.params for l in self.layers ], [])
        print 'Params done'

    def grads(self):
        gs = T.grad(self.cost, self.ps)
        print 'Grads half way'
        self.gs = []
        #for g in self.gs:
        #    g /= g.norm(2)
        #    self.gs.append(T.clip(g, -self.clipping, self.clipping))
        self.gs = [ T.clip(g, -self.clipping, self.clipping) for g in gs ]
        print 'Grads done'

    def updates(self):
        rv = []
        for param, grad in zip(self.ps, self.gs):
            delta = shared_zeros(param.get_value().shape)
            c = self.momentum * delta - self.lr * grad
            rv.append((param, param + c))
            rv.append((delta, c))
        self.updates = rv
        print 'Updates done'

    def make(self):
        self.train = theano.function(
            inputs=[self.input, self.target],
            outputs=[self.cost, self.output],
            updates=self.updates,
            on_unused_input='ignore')
        print 'Training function done'

        self.generate = theano.function(
            inputs=[self.output],
            outputs=self.output,
            on_unused_input='ignore')
        print 'Generating function done'

    def save(self):
        f = file('model.save', 'wb')
        cPickle.dump(self.train, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.generate, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        f = file('model.save', 'rb')
        self.train = cPickle.load(f)
        self.generate = cPickle.load(f)
        f.close()
