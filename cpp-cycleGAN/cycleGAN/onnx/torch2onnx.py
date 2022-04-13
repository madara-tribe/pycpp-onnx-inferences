import torch.onnx 
import sys
sys.path.append('../')
from solver import load_model
from cfg import Cfg

def covert_onnx(): 
    output_name = "cycleGAN_AB.onnx"
    device =torch.device("cpu")
    cfg = Cfg
    G_AB, G_BA, D_A, D_B = load_model(cfg, device, model_path='../tb/')

    dummy_input = torch.randn(1, 3, 256, 256)  
    # Export the model   
    torch.onnx.export(G_AB, dummy_input,
         output_name, verbose=True, input_names = ['input1'],
         output_names = ['output1'])
    print(" ") 
    print('Model has been converted to ' + output_name)

if __name__=='__main__':
    covert_onnx() 
