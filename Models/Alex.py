from Settings import *

class alex_mnist(nn.Module):
    def __init__(self):
        super(alex_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(800, 256)
        self.fc2 = nn.Linear(256, 10)

        self.to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
        
class alex_fmnist(nn.Module):
    def __init__(self):
        super(alex_fmnist, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.05),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 10),
        )

        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class alex_cifar10(nn.Module):
    def __init__(self):
        super(alex_cifar10, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier= nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.05),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 10),
        )

        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)

        return x


