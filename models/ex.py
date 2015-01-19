import nn

network = nn.Container()
network.add(nn.Reshape((1, 784)))
network.add(nn.Linear(784, 100))
network.add(nn.Sigmoid())
network.add(nn.Linear(100, 10))
network.add(nn.Sigmoid())
network.add(nn.MSE(), cost=True)
network.make()