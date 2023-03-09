import torch
import torchvision.datasets as datasets
from torchvision import transforms
from matplotlib import pyplot as plt

BATCH_SIZE = 40

convert_tensor = transforms.ToTensor()

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=convert_tensor)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=convert_tensor)


train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=BATCH_SIZE)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=BATCH_SIZE)
