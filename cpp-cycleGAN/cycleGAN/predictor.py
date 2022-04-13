import cv2, os
import math
import numpy as np
import torch
import torchvision
from cfg import Cfg
from solver import load_model
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

cfg = Cfg
class Predictor():
    def __init__(self, directory, device):
        self.directory = directory
        self.device = device
        
    def predictGenerator(self, img_path):
        inputs=[]
        for idx, imgsp in enumerate(os.listdir(img_path)):
            imgs=cv2.imread(img_path+'/'+imgsp)
            if imgs is not None:
                imgs = cv2.resize(imgs, (256, 256))
                imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
                inputs.append(imgs)
        return (np.array(inputs)/127.5)-1

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[1:4]
        image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
        return image

    def torch_combine_img(self, img_list):
        img1 = torch.cat(img_list[:4], 2)
        img2 = torch.cat(img_list[4:8], 2)
        img3 = torch.cat(img_list[8:12], 2)
        img4 = torch.cat(img_list[12:16], 2)
        combined_img = torch.cat((img1, img2, img3, img4), 1)
        return combined_img
        
    def predict(self):
        domeinB=[]
        imgs = self.predictGenerator('data/trainA')
        G_BA, G_AB, _, _ = load_model(cfg, self.device, model_path='tb')
        for idx, x in tqdm(enumerate(imgs)):
            x = torch.from_numpy(np.expand_dims(x, axis=0)).permute((0, 3, 1, 2))
            output = G_BA(x.to(self.device).float())
            output = make_grid(output, nrow=5, normalize=True)
            save_image(output, os.path.join(self.directory, 'pred_img_{}.png'.format(idx)), normalize=False)
            domeinB.append(output)
            if idx==16-1:
                break
                
        combined_img = self.torch_combine_img(domeinB)
        save_image(combined_img, os.path.join(self.directory, 'combined.png'), normalize=False)
        
if __name__ == '__main__':
    directory = "results"
    os.makedirs(directory, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Predictor(directory, device).predict()
