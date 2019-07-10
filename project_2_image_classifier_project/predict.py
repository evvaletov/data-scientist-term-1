import argparse
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

device = 'cpu'

print('===================================')
print('| Deep Learning Image Recognition |')
print('| Category prediction predict.py  |')
print('===================================')

parser = argparse.ArgumentParser(description='Predict image category using a pre-trained neural network model.')
parser.add_argument('item', help='input image file name')
parser.add_argument('checkpoint_fn', help='model checkpoint file name', default='checkpoint.dat')
parser.add_argument('--gpu', action='store_true', help='use the GPU for computations (default: CPU)')
parser.add_argument('--top_k', type=int, help='number of top categories to display (default: 3)', default=3)
parser.add_argument('--category_names', help='use custom category names mapping (default: cat_to_name.json)',
                    default='cat_to_name.json')
args = parser.parse_args()

if args.gpu:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("GPU is not available. Using CPU.")

if args.top_k < 1:
    print("Number of categories to display cannot be lower than 1")
    sys.exit()


def check_file(file):
    if not os.path.isfile(file):
        print('File ' + file + ' does not exist')
        sys.exit()


for file in [args.item, args.checkpoint_fn, args.category_names]:
    check_file(file)


## Load the checkpoint and rebuild model


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        (
            'fc1',
            nn.Linear(in_features=checkpoint['input_size'], out_features=checkpoint['hidden_layers'][0], bias=True)),
        ('relu1', nn.ReLU()),
        ('do1', nn.Dropout(p=checkpoint['dropouts'][0])),
        ('fc2',
         nn.Linear(in_features=checkpoint['hidden_layers'][0], out_features=checkpoint['output_size'], bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device) + ' for computations')
    model = model.to(device);

    return model


print('Loading model from ' + args.checkpoint_fn)
model = load_checkpoint(args.checkpoint_fn)
model.eval()

## Load label mapping
print('Using ' + args.category_names + ' for category name mapping')
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


## Image processing function    

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.convert('RGB')
    mindim = np.min([im.width, im.height])
    im = im.crop(((im.width - mindim) // 2, (im.height - mindim) // 2, (im.width - mindim) // 2 + mindim,
                  (im.height - mindim) // 2 + mindim))
    im.thumbnail((224, 224))
    np_image = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image


## Prediction function

idx_to_class = {k: v for v, k in model.class_to_idx.items()}


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    with torch.no_grad():
        inputs = process_image(image_path)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.float)
        logps = model.forward(inputs)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.cpu().detach().numpy()[0]
        top_class = top_class.cpu().detach().numpy()[0]
        top_class = [cat_to_name[idx_to_class[x]] for x in top_class]
    return top_p, top_class


top_p, top_class = predict(args.item, model, topk=args.top_k)
print('Showing top ' + str(args.top_k) + ' probable categories')
print('--------------------------')
for j in range(args.top_k):
    print('(' + str(j + 1) + ") " + top_class[j] + ' , probability = ' + str(top_p[j] * 100) + ' %')
