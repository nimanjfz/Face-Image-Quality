import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import model
# from temp import model
from dataset import train_loader
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

loss_fn = nn.MSELoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
writer = SummaryWriter('./logs/runs')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.train()
for epoch in range(20):
    running_loss = 0.0
    loss_values = []
    scheduler.step()
    for i, data in enumerate(tqdm(train_loader), 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        outputs = outputs.reshape(-1)
        outputs = outputs.type(torch.float64)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        running_loss += loss.item()
        if i % 10 == 9:
            loss, current = (running_loss / 10), i * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_loader) * len(images):>5d}]")
            running_loss = 0.0
    print(f"Epoch {epoch + 1}\n-------------------------------")

torch.save(model.state_dict(), "./saved/model-trained-on-ms1mv3.pth")
torch.save(loss_values, "./saved/loss-trained-on-ms1mv3")
plt.plot(loss_values)
plt.show()