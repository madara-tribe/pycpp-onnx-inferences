from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

EPOCHS = 5
BATCH_SIZE = 8
HEIGHT = WIDTH = 299

def load_model(gender_cls=2, age_cls=1):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    input_shape = (HEIGHT, WIDTH, 3)
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=True)
    bottle = base_model.get_layer(index=-3).output
    
    # gender
    x1 = Dropout(rate=0.3)(bottle)
    x1 = GlobalAveragePooling2D()(x1)
    gender_output = Dense(units=gender_cls, activation='softmax', name='gender_output')(x1)

    # age
    x2 = Dropout(rate=0.3)(bottle)
    x2 = GlobalAveragePooling2D()(x2)
    age_output = Dense(units=age_cls, activation='sigmoid', name='age_output')(x2)

    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])  
    model.summary()
    return model
    
if __name__=='__main__':
    model = load_model()
