## Import packages

import json
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models

## Process command line arguments

print('===================================')
print('| Deep Learning Image Recognition |')
print('| Network training tool train.py  |')
print('===================================')

# Set default values
data_dir = 'flowers'
arch = 'vgg19'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
epochs = 10
learning_rate = 0.001
hidden_units = 1024

for item in sys.argv[1:]:
    if item in ['--help', '-h', '/h']:
        print('Usage: python train.py [data_dir] [OPTIONS]')
        print('where OPTIONS can be a combination of:')
        print('   --gpu / --cpu        Use GPU / CPU for computations (default: ' + device + ')')
        print('   --arch ARCH          Use neural network model ARCH (default: ' + arch + ')')
        print('   --learning_rate LR   Use learning rate LR (default: ' + str(learning_rate) + ')')
        print('   --hidden_units HU    Use HU hidden untils it the classifier (default: ' + str(hidden_units) + ')')
        print('   --epochs EPOCHS      Use EPOCHS epochs (default: ' + str(epochs) + ')')
        print('Default data_dir: ' + data_dir)
        sys.exit()

# Iterate over command line arguments
key = None
if len(sys.argv) > 1:
    for item in sys.argv[1:]:
        if item == '--gpu':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                print('CUDA is not available.')
        elif item == '--cpu':
            device = 'cpu'
        elif item.startswith("--"):
            key = item
        elif key == None:
            if os.path.isdir(item):
                data_dir = item
            else:
                print('Directory ' + item + ' does not exist.')
        elif key == '--arch':
            arch = item
            key = None
        elif key == '--learning_rate':
            try:
                learning_rate = float(item)
            except ValueError:
                print("Please provide the learning rate as a floating point number")
            if not learning_rate > 0:
                print("Learning rate must be positive")
                sys.exit()
            key = None
        elif key == '--hidden_units':
            try:
                hidden_units = int(item)
            except ValueError:
                print("The number of hidden units must be an integer")
                sys.exit()
            if not hidden_units > 0:
                print("The number of hidden units must be positive")
                sys.exit()
            key = None
        elif key == '--epochs':
            try:
                epochs = int(item)
            except ValueError:
                print("The number of epochs must be an integer")
                sys.exit()
            if not epochs > 0:
                print("The number of epochs must be positive")
                sys.exit()
            key = None
        else:
            print('Unrecognized command line option ' + key)

if key is not None:
    if key in ['--arch', '--learning_rate', '--hidden_units', '--epochs']:
        print("No value specifed for the command line option " + key)
    else:
        print('Unrecognized command line option ' + key)

print("Data directory: " + data_dir)
print("Model architecture: " + arch)
print("Computation device: " + device)
print("Learning rate: " + str(learning_rate))
print("Hidden units: " + str(hidden_units))
print("Epochs: " + str(epochs))

## Load data

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

testvalid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
datasets = {
    'train': datasets.ImageFolder(train_dir, transform=train_transforms),
    'valid': datasets.ImageFolder(valid_dir, transform=testvalid_transforms),
    'test': datasets.ImageFolder(test_dir, transform=testvalid_transforms)
}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {item: torch.utils.data.DataLoader(datasets[item], batch_size=32, shuffle=True)
               for item in datasets.keys()}

## Load label map

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

## Load model

model = models.__dict__[arch](pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(in_features=25088, out_features=hidden_units, bias=True)),
    ('relu1', nn.ReLU()),
    ('do1', nn.Dropout(p=0.2)),
    ('fc2', nn.Linear(in_features=hidden_units, out_features=102, bias=True)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
lrscheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

model = model.to(device)

## Train the model

print("Starting network training...")
steps = 0
running_loss = 0
print_every = 50
for epoch in range(epochs):
    lrscheduler.step()
    for inputs, labels in dataloaders['train']:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        # print(labels[:10])     
        # print(labels.size(), logps.size())   
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.DoubleTensor)).item()

            print(f"Epoch {epoch + 1}/{epochs}, step {steps}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Validation loss: {test_loss / len(dataloaders['valid']):.3f}.. "
                  f"Validation accuracy: {accuracy / len(dataloaders['valid']):.3f}")
            running_loss = 0
            model.train()

print("Training complete")


## Test the network

def test_network():
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.DoubleTensor)).item()

    print(f"Test loss: {test_loss / len(dataloaders['test']):.3f}.. "
          f"Test accuracy: {accuracy / len(dataloaders['test']):.3f}")


test_network()

## Save the checkpoint

model.class_to_idx = datasets['train'].class_to_idx

model.cpu()
checkpoint = {'arch': arch,
              'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [each.out_features for each in model.classifier
                                if isinstance(each, torch.nn.modules.linear.Linear)],
              'dropouts': [each.p for each in model.classifier
                           if isinstance(each, torch.nn.modules.Dropout)],
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, 'checkpoint.dat')
