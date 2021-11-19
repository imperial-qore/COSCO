import torch
from torchvision import datasets, models, transforms
from PIL import Image
import os
from sys import argv
from tqdm import tqdm

# Reference: https://www.benchcouncil.org/aibench/aiotbench/index.html
# Luo, C., et al. AIoT bench: towards comprehensive benchmarking mobile and embedded device intelligence. 
# In International Symposium on Benchmarking, Measuring and Optimization (pp. 31-35) 2018. Springer, Cham.

transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

## Allowed model names
allowed_models = ['resnet18', 'resnet34', 'squeezenet1_0', 'mobilenet_v2', 'mnasnet1_0', 'googlenet', 'resnext50_32x4d']
multiplier = [2, 1, 4, 2, 1, 3, 1]
multiplier = dict(zip(allowed_models, multiplier))

## Input model
input_model = argv[1]

dataset = os.listdir('assets')

model = eval(f'models.{input_model}(pretrained=True)') 
model.eval()

for _ in tqdm(list(range(multiplier[input_model] * 1000))):
	for fname in dataset:
		img = Image.open('assets/'+fname)
		img_t = transform(img)
		batch_t = torch.unsqueeze(img_t, 0)
