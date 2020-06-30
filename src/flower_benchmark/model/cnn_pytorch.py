import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from typing import Optional, Tuple

# converte cnn from keras to pytorch
class cnn_pytorch(nn.Module):
    def __init__(self):
        super(cnn_pytorch,self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,1)
        self.conv2 = nn.Conv2d(32,64,5,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x,dim=1)
        return output


optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
lr=1e-3
momentum = 0.9
criterion = nn.CrossEntropyLoss()
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    avg_loss = 0
    # in training loop:
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # zero the gradient buffers
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() # Does the update
        avg_loss+= loss.item()
    avg_loss/=len(train_loader.dataset)
    return avg_loss

for epoch in range(1, epochs + 1):
    trn_loss = train(model, device, train_loader, optimizer, epoch, criterion)
    train_losses.append(trn_loss)
    accuracy_list.append(accuracy)