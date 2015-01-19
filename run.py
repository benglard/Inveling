import nn, json, numpy, cv2
from time import time
from random import choice
import theano, subprocess

path = './data/pascal1k/'

with open(path + 'data.json') as infile:
    data = json.load(infile)
    sents = sum([ s for n, s in data.iteritems() ], [])
    words = set()
    for s in sents:
        for word in s.split():
            words.add(word)
    words_lst = list(words)
    dict_size = len(words)

def sent2matrix(sent):
    tokens = sent.split()
    arr = numpy.zeros((len(tokens), dict_size), dtype=theano.config.floatX)
    for idx, word in enumerate(tokens):
        arr[idx, words_lst.index(word)] = 1
    return arr

def get_data():
    for filename, sents in data.iteritems():
        sent = choice(sents)
        mat = sent2matrix(sent)
        img = cv2.imread(path + filename)
        rs = cv2.resize(img, (100, 100)).reshape((300, 100)).astype(theano.config.floatX)
        yield (mat, rs)

network = nn.Container()
# Encoder
network.add(nn.LSTM(dict_size, 300, 300))
network.add(nn.LSTM(300, 100, 100))
network.add(nn.LSTM(100, 100, 33))
# Decoder
network.add(nn.Reshape((1, 3, 11, 100)))
network.add(nn.SpatialConvolution((1, 3, 11, 100), (16, 3, 5, 5)))
network.add(nn.ReLU())
network.add(nn.SpatialMaxPooling((2, 2), 48))
network.add(nn.Reshape((2304,)))
network.add(nn.Linear(2304, 30000))
network.add(nn.Reshape((300, 100)))
network.add(nn.MSE(), cost=True)
print 'Network created'
print 'Compiling function'
network.make()
print 'Function created'

print 'In training'
k = 0
#n_train = 900
n_epochs = 5
try:
    for n in xrange(n_epochs):
        for x, y in get_data():
            s = time()
            cost, output = network.train(x, y)
            print cost, k, time() - s
            #cv2.imwrite('./data/{}Y.png'.format(k), y.reshape((100, 100, 3)))
            #cv2.imwrite('./data/{}P.png'.format(k), output.reshape((100, 100, 3)))
            k += 1
except:
    pass
finally:
    #network.save()
    pass

while True:
    query = raw_input('Enter a string: ')
    mat = sent2matrix(query)
    #fake = numpy.random.randn(300, 100).astype(theano.config.floatX)
    #output = network.train(mat, fake)[1].reshape((100, 100, 3))
    output = network.generate(mat).reshape((100, 100, 3))
    #cv2.imwrite('out.png', output)
    #cv2.imwrite('blur.png', cv2.medianBlur(output, 5))
    #subprocess.call('display out.png', shell=True)
    #subprocess.call('display blur.png', shell=True)