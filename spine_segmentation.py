import os
import numpy as np
from PIL import Image
import sys
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision.models.segmentation import fcn_resnet50
from zipfile import ZipFile 


class Dataset(BaseDataset):
    """Creates a map dataset that yields (image,mask) pairs to data loader"""
    def __init__(self, images_dir, masks_dir):
        self.ids = os.listdir(images_dir)
        self.ids = [s.partition('.')[0] for s in self.ids]
        self.images = [os.path.join(images_dir, id+'.png') for id in self.ids]
        self.masks = [os.path.join(masks_dir, id+'.npy') for id in self.ids] 
        
    def __getitem__(self, i):
        image = Image.open(self.images[i])
        image = np.asarray(image)
        image = np.tile(image,(3,1,1)).astype(np.float32)
        mask = np.load(self.masks[i])
        return image, mask

    def __len__(self): 
        return len(self.ids)

class loss_2d(nn.Module):
    """Defines a cross entropy based loss function between 2-d predictions and masks for three classes""" 
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, prediction, mask):
        prediction = prediction.reshape((3,-1))
        prediction = torch.t(prediction) 
        mask = mask.view(-1)
        return self.loss(prediction, mask.long())

def train(train_dataset, model, optimizer):
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    train_loss = 0
    loss_func = loss_2d()
    for sample in loader:
        optimizer.zero_grad()
        y = model(sample[0])
        loss = loss_func(y['out'], sample[1])
        train_loss += loss.item()*len(sample)
        loss.backward()
        optimizer.step()
    # Accumulated loss is averaged over all of the samples in train dataset in each epoch 
    train_loss /= len(loader)
    return train_loss, y
 
def main():
    if __name__ == "__main__":
        # Extracting data from the given file
        file_name = "coding_challenge.zip"
        with ZipFile(file_name, 'r') as f:
            f.extractall()
        path = "coding_challenge"
        images_path = path + "/images"
        masks_path = path + "/masks"
        train_dataset = Dataset(images_path, masks_path)
        model = fcn_resnet50(pretrained=False, progress=True, num_classes=3, aux_loss=None) 
        optimizer = SGD(model.parameters(),lr=0.01, momentum=0.9)
        n_epochs = 3
        for i in range(n_epochs):
            loss, pred = train(train_dataset, model, optimizer)
            print(f"Training loss at epoch {i+1} : {loss}")

main()
