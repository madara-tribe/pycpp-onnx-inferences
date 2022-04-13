from solver import Trainer
from cfg import Cfg
import sys, os
import torch

def main(cfg, model_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    
    # To speed up training
    device = torch.device("cuda")
    num_workers = 8 if os.cpu_count() > 8 else os.cpu_count()
    print(num_workers)
    pin_memory = True
    train = Trainer(cfg, num_workers=num_workers, pin_memory=pin_memory)
    train.start_training(cfg=cfg, device=device, model_path=model_path)
    
          
if __name__ == '__main__':
    if len(sys.argv)>1:
        model_path = 'tb'
    else:
        model_path = None
    cfg = Cfg
    main(cfg, model_path)


