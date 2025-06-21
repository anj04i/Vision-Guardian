import os
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

# Create directory for model if not exists
if not os.path.exists('places365'):
    os.mkdir('places365')

# Download the model if not present
model_file = 'places365/resnet18_places365.pth.tar'
if not os.path.exists(model_file):
    print("Downloading Places365 model...")
    url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
    urllib.request.urlretrieve(url, model_file)

# Load label names
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    categories_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    urllib.request.urlretrieve(categories_url, file_name)

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])

# Define model
def load_model():
    model = models.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, classes
