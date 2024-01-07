import torch
import torch.nn as nn
import torch.optim as optim

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def fit(self, x, y, epochs=1, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    def predict(self, x):
        return self.forward(x)
    

    def save(self):
        name = str(hash(self))
        torch.save(self.state_dict(), "models/{}.pth".format(name))

    def load(self, path):
        self.load_state_dict(torch.load(path))
