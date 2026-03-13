from sklearn.datasets import make_moons
import torch
from torch import Tensor
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F

class TwoMoons:
    def __init__(self, batch_size, n_batches=1000, noise=0.1):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.noise = noise

    def __iter__(self):
        for _ in range(self.n_batches):
            x, y = make_moons(self.batch_size, noise=self.noise)
            yield Tensor(x), Tensor(y)

    def __len__(self):
        return self.n_batches

class ChessBoard:
    def __init__(self, batch_size, n_batches=2000):
        self.n_batches = n_batches
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            x1 = torch.rand(self.batch_size) * 4 - 2
            x2_ = torch.rand(self.batch_size) - torch.randint(high=2, size=(self.batch_size, )) * 2
            x2 = x2_ + (torch.floor(x1) % 2)
            data = torch.cat([x1[:, None], x2[:, None]], dim=1)
            label = (torch.floor(x1) % 2) + 2*(torch.abs(x2) >= 1) + 4*(x1 >= 0)
            yield data.float(), F.one_hot(label.long(), num_classes=8)

    def __len__(self):
        return self.n_batches

class MNIST:
    def __init__(self, batch_size):
        transform = transforms.Compose([
            transforms.Resize((24, 24)),
            transforms.ToTensor()
        ])
        self.dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
        )
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def __iter__(self):
        for x,y in iter(self.loader):
            yield x, F.one_hot(y.long(), num_classes=10)

    def __len__(self):
        return len(self.loader)
    
class OneImg:
    def __init__(self, batch_size, img='data/FML.png', n_batches=500, noise=0.1):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.noise = noise
        self.transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
        self.img = self.transform(Image.open(img))

    def __iter__(self):
        for _ in range(self.n_batches):
            yield torch.randn(self.batch_size, *self.img.shape) * self.noise + self.img, None

    def __len__(self):
        return self.n_batches