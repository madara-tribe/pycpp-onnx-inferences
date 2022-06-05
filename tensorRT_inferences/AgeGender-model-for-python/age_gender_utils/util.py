import numpy as np
from tensorflow.keras.utils import to_categorical

gender_cls = 2
generation_cls=6
identity_cls=21

def preprocess(genders, generation, identity):
    genders = to_categorical(genders, num_classes = gender_cls)
    generation = to_categorical(generation, num_classes = generation_cls)
    identity = to_categorical(identity, num_classes = identity_cls)
    return genders, generation, identity

