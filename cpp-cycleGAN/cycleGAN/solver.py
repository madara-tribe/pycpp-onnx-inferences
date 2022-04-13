import os, math
import numpy as np
import itertools
import datetime
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from fastprogress import progress_bar

from models.cycleGAN import Discriminator, GeneratorResNet,weights_init_normal
from models.utils import *
from DataLoders import RootDataLoders
import torch.nn as nn
import torch.nn.functional as F
import torch
        
        
def load_model(cfg, device, model_path=None):
    input_shape = (3, cfg.img_height, cfg.img_width)
    
    G_AB = GeneratorResNet(input_shape, cfg.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, cfg.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)
    if device:
        G_AB = G_AB.to(device)
        G_BA = G_BA.to(device)
        D_A = D_A.to(device)
        D_B = D_B.to(device)

    if model_path:
        # Load pretrained models
        G_AB.load_state_dict(torch.load(os.path.join(model_path, "G_AB.pth"), map_location=device))
        G_BA.load_state_dict(torch.load(os.path.join(model_path, "G_BA.pth"), map_location=device))
        D_A.load_state_dict(torch.load(os.path.join(model_path, "D_A.pth"), map_location=device))
        D_B.load_state_dict(torch.load(os.path.join(model_path, "D_B.pth"), map_location=device))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)
    return G_AB, G_BA, D_A, D_B
    
    
class Trainer():
    def __init__(self, cfg, num_workers, pin_memory):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.ck = cfg.ck
        self.tb = cfg.tb
        if not os.path.exists(self.ck):
            os.makedirs(self.ck, exist_ok=True)
            os.makedirs(self.tb, exist_ok=True)
            
        dataset = RootDataLoders(cfg.train_path_imageA, cfg.train_path_imageB)
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        
        # Test data loader
        dataset = RootDataLoders(cfg.val_path_imageA, cfg.val_path_imageB)
        self.val_dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=1)
        self.writer = SummaryWriter(log_dir=cfg.TENSORBOARD_DIR,
                           filename_suffix=f'OPT_LR_{cfg.lr}_BS_Size_{cfg.img_width}',
                           comment=f'OPT_LR_{cfg.lr}_BS_Size_{cfg.img_width}')
        
    def model_init(self, cfg, G_AB, G_BA, D_A, D_B):
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=cfg.lr, betas=(cfg.b1, cfg.b2)
        )
        self.optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))
        self.optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=cfg.lr, betas=(cfg.b1, cfg.b2))

        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(cfg.n_epochs, cfg.epoch, cfg.decay_epoch).step
        )
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=LambdaLR(cfg.n_epochs, cfg.epoch, cfg.decay_epoch).step
        )
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=LambdaLR(cfg.n_epochs, cfg.epoch, cfg.decay_epoch).step
        )


    def sample_images(self, G_AB, G_BA, real_A, real_B, itr, epoch):
        imgs = next(iter(self.val_dataloader))
        G_AB.eval()
        G_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = G_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 2)
        save_image(image_grid, os.path.join(self.ck, '{}_{}'.format(itr, epoch)+'_images.png'), normalize=False)

    def start_training(self, cfg, device, model_path=None):
        global Tensor
        Tensor = torch.cuda.FloatTensor if device else torch.Tensor
        ### To spped up training
        torch.backends.cudnn.benchmark =True
        if device:
            self.criterion_GAN.to(device)
            self.criterion_cycle.to(device)
            self.criterion_identity.to(device)
        # Buffers of previously generated samples
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()
        
        G_AB, G_BA, D_A, D_B = load_model(cfg, device, model_path=model_path)
        self.model_init(cfg, G_AB, G_BA, D_A, D_B)
        
        prev_time = time.time()
        cur = 0
        for epoch in range(cfg.epoch, cfg.n_epochs):
            i = 0
            for batch in progress_bar(self.dataloader):
                cur += 1
                i += 1
                # Set model input
                real_A = Variable(batch["A"].type(Tensor))
                real_B = Variable(batch["B"].type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                G_AB.train()
                G_BA.train()

                self.optimizer_G.zero_grad()

                # Identity loss
                loss_id_A = self.criterion_identity(G_BA(real_A), real_A)
                loss_id_B = self.criterion_identity(G_AB(real_B), real_B)

                loss_identity = (loss_id_A + loss_id_B) / 2

                # GAN loss
                fake_B = G_AB(real_A)
                loss_GAN_AB = self.criterion_GAN(D_B(fake_B), valid)
                fake_A = G_BA(real_B)
                loss_GAN_BA = self.criterion_GAN(D_A(fake_A), valid)

                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                # Cycle loss
                recov_A = G_BA(fake_B)
                loss_cycle_A = self.criterion_cycle(recov_A, real_A)
                recov_B = G_AB(fake_A)
                loss_cycle_B = self.criterion_cycle(recov_B, real_B)

                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # Total loss
                loss_G = loss_GAN + cfg.lambda_cyc * loss_cycle + cfg.lambda_id * loss_identity

                loss_G.backward()
                self.optimizer_G.step()

                # -----------------------
                #  Train Discriminator A
                # -----------------------

                self.optimizer_D_A.zero_grad()

                # Real loss
                loss_real = self.criterion_GAN(D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = self.criterion_GAN(D_A(fake_A_.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2

                loss_D_A.backward()
                self.optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------

                self.optimizer_D_B.zero_grad()

                # Real loss
                loss_real = self.criterion_GAN(D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = self.criterion_GAN(D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2

                loss_D_B.backward()
                self.optimizer_D_B.step()

                loss_D = (loss_D_A + loss_D_B) / 2

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(self.dataloader) + i
                batches_left = cfg.n_epochs * len(self.dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                self.writer.add_scalar("loss_D", loss_D.item(), cur)
                self.writer.add_scalar("loss_G", loss_G.item(), cur)
                self.writer.add_scalar("loss_GAN", loss_GAN.item(), cur)
                self.writer.add_scalar("loss_cycle", loss_cycle.item(), cur)
                self.writer.add_scalar("loss_identity", loss_identity.item(), cur)
                # If at sample interval save image
                if cur % cfg.sample_interval == 0:
                    self.sample_images(G_AB, G_BA, real_A, real_B, cur, epoch)
            print(epoch, loss_D.item(), loss_G.item(), loss_GAN.item())
            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()
            
            torch.save(G_AB.state_dict(), os.path.join(self.tb, "G_AB.pth"))
            torch.save(G_BA.state_dict(), os.path.join(self.tb, "G_BA.pth"))
            torch.save(D_A.state_dict(), os.path.join(self.tb, "D_A.pth"))
            torch.save(D_B.state_dict(), os.path.join(self.tb, "D_B.pth"))

