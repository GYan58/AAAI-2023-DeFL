from Settings import *


class fnn_mnist(nn.Module):
    def __init__(self):
        super(fnn_mnist, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

        self.to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class fnn_fmnist(nn.Module):
    def __init__(self):
        super(fnn_fmnist, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

        self.to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class fnn_cifar10(nn.Module):
    def __init__(self):
        super(fnn_cifar10, self).__init__()
        self.fc1 = nn.Linear(3072, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

        self.to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


