import theano
import numpy

def shared_normal(size, scale=1):
    return theano.shared(
        numpy.random.normal(size=size, scale=scale).astype(theano.config.floatX))

def shared_zeros(shape):
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))