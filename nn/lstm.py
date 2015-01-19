from nn.utils import shared_normal, shared_zeros
import theano
import theano.tensor as T
from math import sqrt
sigmoid = T.nnet.sigmoid
tanh = T.tanh
dot = T.dot

class LSTM(object):

    def __init__(self, n_in, n_hidden, n_out, scale=0.5):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.scale = sqrt(1.0/n_in)
        self.scale_out = sqrt(1.0/n_hidden)

        self.W_hi = shared_normal((n_hidden, n_in), scale=self.scale)
        self.W_ci = shared_normal((n_hidden, n_hidden), scale=self.scale)
        self.b_i = shared_zeros((n_hidden,))
        self.W_hf = shared_normal((n_hidden, n_in), scale=self.scale)
        self.W_cf = shared_normal((n_hidden, n_hidden), scale=self.scale)
        self.b_f = shared_zeros((n_hidden,))
        self.W_hc = shared_normal((n_hidden, n_in), scale=self.scale)
        self.b_c = shared_zeros((n_hidden,))
        self.W_ho = shared_normal((n_hidden, n_in), scale=self.scale)
        self.W_co = shared_normal((n_hidden, n_hidden), scale=self.scale)
        self.b_o = shared_zeros((n_hidden,))
        self.W_od = shared_normal((n_out, n_hidden), scale=self.scale_out) #output decoder
        self.b_od = shared_zeros((n_out, n_hidden))

        self.params = [ self.W_hi, self.W_ci, self.b_i,
                        self.W_hf, self.W_cf, self.b_f,
                        self.W_hc, self.b_c,
                        self.W_ho, self.W_co, self.b_o,
                        self.W_od, self.b_od ]

    def forward(self, input):
        # Modeled after
        # https://www.cs.toronto.edu/~hinton/absps/RNN13.pdf
        # Equations 3-7

        def step(prev_h, prev_c):
            i_t = sigmoid(dot(self.W_hi, prev_h) + dot(self.W_ci, prev_c) + self.b_i) # (3)
            f_t = sigmoid(dot(self.W_hf, prev_h) + dot(self.W_cf, prev_c) + self.b_f) # (4)
            c_t = f_t * prev_c + i_t * tanh(dot(self.W_hc, prev_h) + self.b_c)        # (5)
            o_t = sigmoid(dot(self.W_ho, prev_h) + dot(self.W_co, prev_c) + self.b_o) # (6)
            h_t = o_t * tanh(c_t)                                                     # (7)
            return h_t, c_t

        mem = shared_zeros((self.n_hidden,))
        (out, _), _ = theano.scan(fn=step, sequences=[input, mem])
        final = out[-1]
        return dot(self.W_od, final) + self.b_od
