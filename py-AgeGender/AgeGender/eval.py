import numpy as np
import glob, os
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from age_gender_utils import load, util
from models.age_gender_model import load_model

EPOCHS = 10
BATCH_SIZE = 8
HEIGHT = WIDTH = 299
WEIGHT_DIR = 'weights'
os.makedirs(WEIGHT_DIR, exist_ok=True)
#ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}


    
def load_dataset():
    print('load test data')
    X_val, gen_val, generation_val, identity_val = load.load_age_gender_data(path="../UTKDATA/part3", img_size=WIDTH)
    gen_val, generation_val, identity_val = util.preprocess(gen_val, generation_val, identity_val)
    X_val = np.array(X_val, dtype='float32')/255
    print(X_val.shape, gen_val.shape, generation_val.shape, identity_val.shape)
    
    return X_val, generation_val, identity_val, gen_val





def eval(weight_path=None):
    X_val, generation_val, identity_val, gen_val = load_dataset()
    model = load_model()
    if weight_path is not None:
        print('weight loading....')
        model.load_weights(os.path.join(WEIGHT_DIR, weight_path))


    acc = model.evaluate(x=X_val, y=[generation_val, identity_val, gen_val])
    print('generation acc:{0}, identity accuracy: {1}, gender accuracy: {2}'.format(acc[4], acc[5], acc[6]))

if __name__=='__main__':
    weight_path = 'cp-age-gender_05.hdf5'
    eval(weight_path=weight_path)
