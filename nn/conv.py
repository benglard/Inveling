import numpy
from nn.utils import shared_normal, shared_zeros
from math import sqrt
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class SpatialConvolution(object):

    def __init__(self, image_shape, filter_shape):
        self.image_shape = image_shape
        self.filter_shape = filter_shape

        scale = sqrt(1.0/numpy.prod(self.filter_shape[1:]))
        self.W = shared_normal(self.filter_shape, scale=scale)
        self.params = [ self.W ]
        self.cost = 0

    def forward(self, input):
        return conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape
        )

class SpatialMaxPooling(object):

    def __init__(self, pool_shape, n_filters):
        self.pool_shape = pool_shape
        self.b = shared_zeros((n_filters,))
        self.params = [ self.b ]
        self.cost = 0

    def forward(self, input):
        return downsample.max_pool_2d(
            input=input,
            ds=self.pool_shape,
            ignore_border=True
        ) + self.b
