import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST
import torch

def collate_fn(data):
  imgs = torch.from_numpy(
    np.array([d[0] for d in data])
  )
  #labels = torch.nn.functional.one_hot(
  #  torch.from_numpy(np.array([ d[1] for d in data ]))
  #)
  labels = torch.tensor([d[1] for d in data])
  return imgs, labels

class MNISTDataset(data.Dataset):
  def __init__(self, dataset):
    self.dataset = dataset
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    return self.dataset[idx][0].astype("float32"), self.dataset[idx][1]
  

  

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic))


mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False, transform=FlattenAndCast())

TrainSet = MNISTDataset(mnist_dataset)
TestSet = MNISTDataset(mnist_dataset_test)



TrainLoader = data.DataLoader(TrainSet, batch_size=100, shuffle=True, collate_fn=collate_fn)
TestLoader = data.DataLoader(TestSet, batch_size=100, shuffle=True, collate_fn=collate_fn)


