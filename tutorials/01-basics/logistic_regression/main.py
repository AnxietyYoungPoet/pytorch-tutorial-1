import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

# Hyper-parameters 
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 1e-4

# MNIST dataset (images and labels)
<<<<<<< HEAD
train_dataset = torchvision.datasets.MNIST(root='/Users/sunpeiquan/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/Users/sunpeiquan/data',
                                          train=False,
=======
train_dataset = torchvision.datasets.MNIST(root='/home/sunpq/datasets', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/home/sunpq/datasets', 
                                          train=False, 
>>>>>>> b0b9947de111390f9f777917d5183ad52f4ab56d
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Logistic regression model
class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU()
    )
    self.mu = nn.Linear(1024, 256)
    self.sigma = nn.Linear(1024, 256)
    self.decoder = nn.Linear(256, num_classes)

  def forward(self, x):
    hidden = self.model(x)
    mu = self.mu(hidden)
    sigma = self.sigma(hidden)
    std = nn.functional.softplus(sigma)
    dist = Normal(mu, std)
    z = dist.rsample()
    return z, dist, mu

encoder = Encoder()

# model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)

        # Forward pass
        z, dist, mu = encoder(images)
        prior = Normal(torch.zeros_like(z), torch.ones_like(z))
        kl = 0.001 * kl_divergence(dist, prior).sum(1).mean()
        loss = criterion(z, labels) + kl

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0.
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        _, _, outputs = encoder(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(encoder.state_dict(), 'model.ckpt')
