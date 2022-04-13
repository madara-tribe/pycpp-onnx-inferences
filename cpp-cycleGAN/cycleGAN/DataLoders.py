import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from cfg import Cfg

cfg = Cfg
transforms_ = [
    transforms.Resize(int(cfg.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((cfg.img_height, cfg.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class RootDataLoders(Dataset):
    def __init__(self, pathA,pathB,train = True):
        self.transform = transforms.Compose(transforms_)

        self.train = train
        self.fileA = sorted(glob.glob(pathA))
        self.fileB = sorted(glob.glob(pathB))
        #self.fileA = self.fileA[:120]
        #self.fileB = self.fileB[:120]
        
    def __getitem__(self, index):
        image_A = Image.open(self.fileA[index % len(self.fileA)])
        image_B = Image.open(self.fileB[index % len(self.fileB)])

        # Convert grayscale images to rgb
        image_A = to_rgb(image_A)
        image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.fileA), len(self.fileB))
        
        

        

