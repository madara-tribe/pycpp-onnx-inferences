from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

EPOCHS = 5
BATCH_SIZE = 8
HEIGHT = WIDTH = 299

def load_model(gender_cls=2, generation_cls=6, identity_cls=21):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    input_shape = (HEIGHT, WIDTH, 3)
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=True)
    bottle = base_model.get_layer(index=-3).output
    bottle = Dropout(rate=0.3)(bottle)
    bottle = GlobalAveragePooling2D()(bottle)
    # gender
    gender_output = Dense(units=gender_cls, activation='softmax', name='gender_output')(bottle)
    
    # generation
    generation_output = Dense(units=generation_cls, activation='softmax', name='generation_output')(bottle)
    
    # identity age
    identity_outout = Dense(units=identity_cls, activation='softmax', name='identity_outout')(bottle)

    model = Model(inputs=base_model.input, outputs=[generation_output, identity_outout, gender_output])
    model.compile(optimizer=adam,
                  loss={'generation_output': 'categorical_crossentropy',
                      'identity_outout': 'categorical_crossentropy',
                      'gender_output': 'binary_crossentropy'},
                  #loss_weights={'realage_output': 1, 'gender_output': 10},
                  metrics={'generation_output':'accuracy',
                           'identity_outout':'accuracy',
                           'gender_output': 'accuracy'})

    model.summary()
    return model

if __name__=='__main__':
    model = load_model()
