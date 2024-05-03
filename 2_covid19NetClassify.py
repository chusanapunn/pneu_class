import torch
import numpy as np

import torchvision
from torch.utils.data import DataLoader
from torch.backends import mps
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn 
from net_covid19 import Covid19Net
import time

def imshow(img, title=None):
  ''' function to show image '''
  plt.imshow(img.permute(1, 2, 0))
  if title is not None:
    plt.title(title)
  plt.show()

# setup some path variables
project_path = './'
data_path = project_path + 'data/Pneumonia/X-Ray/'

# select device to run the computations on
if mps.is_available(): # MAcOS with Metal support
    device = torch.device('mps')
elif torch.cuda.is_available(): # Nvidia GPU
    device = torch.device('cuda')
else: # CPU
    device = torch.device('cpu')

# device = torch.device('cpu') # uncomment this line to run on CPU
print(20*"#")
print("Device used: ", device)
print(20*"#")
# Define the transformations
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # Resize the image to 224x224
    torchvision.transforms.ToTensor() # Convert the image to a pytorch tensor
])

# Define train / test dataset
train_dataset = torchvision.datasets.ImageFolder(data_path + 'train/', transform=image_transform)
test_dataset = torchvision.datasets.ImageFolder(data_path + 'test/', transform=image_transform)

# Check the classes labels
class_labels = train_dataset.classes
print(class_labels)

# Check the number of samples in the train and test dataset
print('Number of images in train set:', len(train_dataset))
print('Number of images in test set:', len(test_dataset))

# # Show sample image
# sample_idx = 240
# sample_image, sample_label = train_dataset[sample_idx]
# print(f'Image Shape: {sample_image.shape}')
# print(f'Label: {class_labels[sample_label]}')

# imshow(sample_image)
# # sample_image[0,:,:]

# # Define the data loader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# get a batch of images and labels
images, labels = next(iter(train_loader))
print(f'Shape of image tensors: {images.shape}')
print(f'Shape of label tensors: {labels.shape}')

# Display the batch of images
class_labels_string = ', '.join([class_labels[label] for label in labels]) # Create a string of class labels indexed by labels
# imshow(torchvision.utils.make_grid(images), title = class_labels_string)
# plt.title(class_labels_string)

print(class_labels_string)
print([class_labels[label] for label in labels])

# Define the model
net = Covid19Net()
print(net)

net.to(device) # move the model to the device

# define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0025, momentum=0.9)


num_epochs = 20 # loop over the dataset multiple times

start_time = time.time()
for epoch in range(num_epochs): # one epoch is a complete pass through the train dataset
    epoch_loss = 0.0
    for batch_index, data in enumerate(train_loader):
        images, labels = data # get the inputs; data is a list of [inputs, labels]
        # inputs.shape, labels.shape
        # images = images[:,0:2]
        # labels = labels[:,0]
        images, labels = images.to(device), labels.to(device) # move the data to the device
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(images)
        # outputs = outputs[:,0]
        # print(outputs.shape)
        # print(labels.shape)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item()

    print(f'Epoch: {epoch}, Loss: {epoch_loss}') # print the loss every epoch

end_time = time.time()
execution_time = end_time - start_time
print(f"Training completed in {execution_time} seconds")


# store the model
torch.save(net.state_dict(), project_path + 'covid19_net.pth')

# reload saved model
net = Covid19Net()
net.load_state_dict(torch.load(project_path + 'covid19_net.pth'))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))
print('Training done on device:', device)
