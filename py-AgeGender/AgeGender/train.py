import numpy as np
import glob, os
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from age_gender_utils import load, util
from models.age_gender_model import load_model

EPOCHS = 1
BATCH_SIZE = 8
HEIGHT = WIDTH = 299
WEIGHT_DIR = 'weights'
os.makedirs(WEIGHT_DIR, exist_ok=True)
#ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}


    
def load_dataset():
    print('load train data')
    X, x_gen, x_generation, x_identity = load.load_age_gender_data(path="../../UTKFace", img_size=WIDTH)
    X_train = X
    y_gen = x_gen
    y_generation = x_generation 
    y_identity = x_identity
    X_train = np.array(X_train, dtype='float32')/255
    y_gen, y_generation, y_identity = util.preprocess(y_gen, y_generation, y_identity)
    print(X_train.shape, y_gen.shape, y_generation.shape, y_identity.shape)
    print(X_train.min(), X_train.max())
    
    y_label = [y_generation, y_identity, y_gen]
    
    print('load val data')
    X_val, gen_val, generation_val, identity_val = load.load_age_gender_data(path="../../UTKFace", img_size=WIDTH)
    gen_val, generation_val, identity_val = util.preprocess(gen_val, generation_val, identity_val)
    X_val = np.array(X_val, dtype='float32')/255
    print(X_val.shape, gen_val.shape, generation_val.shape, identity_val.shape)
    
    y_val = [generation_val, identity_val, gen_val]
    return X_train, y_label, X_val, y_val



def create_callbacks():
    checkpoint_path = os.path.join(WEIGHT_DIR, "cp-age-gender_{epoch:02d}.hdf5")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    target_monitor = 'val_loss'
    cp_callback = ModelCheckpoint(checkpoint_path, monitor=target_monitor, verbose=1, save_best_only=True, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, verbose = 1)
    calllbacks = [reduce_lr, cp_callback]
    return calllbacks


def train(weight_path=None):
    X_train, y_label, X_val, y_val = load_dataset()
    model = load_model()
    if weight_path is not None:
        model.load_weights(os.path.join(WEIGHT_DIR, weight_path))

    calllback = create_callbacks()

    print('train')
    startTime1 = datetime.now()
    hist1 = model.fit(x=X_train, y=y_label, batch_size=2, epochs=EPOCHS, validation_data=(X_val, y_val), verbose=1, callbacks=calllback)

    endTime1 = datetime.now()
    diff1 = endTime1 - startTime1
    print("\n")
    print("Elapsed time for Keras training (s): ", diff1.total_seconds())
    print("\n")

    for key in ["loss", "val_loss"]:
        plt.plot(hist1.history[key],label=key)
    plt.legend()

    plt.savefig(os.path.join(WEIGHT_DIR, "model" + str(EPOCHS) + "_training_curves_" + str(WIDTH) + "x" + str(HEIGHT) + ".png"))

    model.save(os.path.join(WEIGHT_DIR, "ep" + str(EPOCHS) + "age_gender" + "_" + str(WIDTH) + "x" + str(HEIGHT) + ".hdf5"))

    acc = model.evaluate(x=X_val, y=y_val)
    print('generation acc:{0}, identity accuracy: {1}, gender accuracy: {2}'.format(acc[4], acc[5], acc[6]))

if __name__=='__main__':
    weight_path = None
    train(weight_path=weight_path)


