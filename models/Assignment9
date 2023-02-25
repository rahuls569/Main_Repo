import torch
import torch.nn as nn
import torch.nn.functional as F

class ULTIMUS(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ULTIMUS, self).__init__()
        self.K = nn.Linear(input_dim, output_dim)
        self.Q = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mat1 = torch.matmul(x, self.K.weight.t())
        mat2 = torch.matmul(x, self.Q.weight.t())
        mat3 = torch.matmul(x, self.V.weight.t())
        am = F.softmax(torch.matmul(mat2, mat1.t()) / (mat1.shape[1] ** 0.5), dim=-1)
        z = torch.matmul(am, mat3)
        out = F.linear(z, self.K.weight.t())
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ultimus1 = ULTIMUS(48, 8)
        self.ultimus2 = ULTIMUS(48, 8)
        self.ultimus3 = ULTIMUS(48, 8)
        self.ultimus4 = ULTIMUS(48, 8)
        self.out = nn.Linear(48, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)
        x = self.out(x)
        return x
