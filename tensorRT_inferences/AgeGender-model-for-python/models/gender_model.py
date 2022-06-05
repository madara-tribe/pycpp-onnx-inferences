from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras import losses
EPOCHS = 5
BATCH_SIZE = 8
HEIGHT = WIDTH = 299

def custom_loss(y_true, y_pred):
    generation_loss = losses.categorical_crossentropy(y_true[0], y_pred[0])
    identity_loss = losses.categorical_crossentropy(y_true[1], y_pred[1])
    gender_loss = losses.binary_crossentropy(y_true[2], y_pred[2])

    return generation_loss, identity_loss, gender_loss
 

def load_model(gender_cls=2, generation_cls=6, identity_cls=21):
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    input_shape = (HEIGHT, WIDTH, 3)
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=True)
    bottle = base_model.get_layer(index=-3).output
    bottle = Dropout(rate=0.3)(bottle)
    bottle = GlobalAveragePooling2D()(bottle)
    # gender
    gender_output = Dense(units=gender_cls, activation='softmax', name='gender_output')(bottle)

    model = Model(inputs=base_model.input, outputs=gender_output)
    model.compile(optimizer=adam,
                  loss={'gender_output': 'binary_crossentropy'},
                  loss_weights={'gender_output': 10},
                  metrics={'gender_output': 'accuracy'})
    model.summary()
    return model

if __name__=='__main__':
    model = load_model()

