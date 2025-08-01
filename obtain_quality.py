from torch.utils.data import Dataset, DataLoader
import glob
import random
import numpy as np
import os
import cv2
from torchvision.transforms import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import model
import torch.nn as nn
import scipy.stats as st
import torch
import pandas as pd

# Define Dataset class
class Dataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        image_path = self.img_dir[idx]
        image = cv2.imread(image_path)
        image = transform(image)
        # label = self.img_labels[idx]
        return image, image_path

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
df = pd.read_csv('./dataset/test/img_test.list', header=None)
img_list = df.values.tolist()
xss = img_list
img_list = [x for xs in xss for x in xs]
dataset = Dataset(img_list, transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("./saved/model-trained-on-ms1mv3.pth"))

quality_scores = torch.empty(0)
quality_scores = quality_scores.to(device)
image_paths_list = []
model.eval()
with torch.no_grad():
    for data in tqdm(data_loader):
        images, image_paths = data
        images = images.to(device)
        image_paths = list(image_paths)
        outputs = model(images)
        outputs = outputs.reshape(-1)
        outputs = outputs.to(device)
        quality_scores = torch.cat((quality_scores, outputs))
        image_paths_list.append(image_paths)


quality_scores = quality_scores.cpu().detach().tolist()
xss = image_paths_list
image_paths_list = [x for xs in xss for x in xs]

quality_scores = pd.DataFrame({'file_names': image_paths_list,
                               'quality_score': quality_scores})
quality_scores.to_csv('./saved/quality_scores_for_test_images.csv', index=False)