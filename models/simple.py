# Encoder
network.add(nn.LSTM(4708, 400, 400))
network.add(nn.LSTM(400, 100, 100))
network.add(nn.LSTM(100, 100, 100 * 3))
# Decoder
network.add(nn.Reshape((3*100, 100)))
network.add(nn.MSE(), cost=True)
