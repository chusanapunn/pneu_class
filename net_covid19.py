import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu

class Covid19Net(nn.Module):
  '''Models a simple Convolutional Neural Network'''

  def __init__(self, num_classes=2):
    super(Covid19Net, self).__init__()
    
    # convolutional layer 1 & max pool layer 1
    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2))
    
    # convolutional layer 2 & max pool layer 2
    self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2))
    
    #Fully connected layer
    self.fc = nn.Linear(32*54*54, num_classes)

  # Feed forward the network
  def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = out.reshape(out.size(0), -1)
      out = self.fc(out)
      return out

net = Covid19Net()
print(net)