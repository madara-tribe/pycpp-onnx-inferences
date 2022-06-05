import os
import sys
sys.path.append('../')
import tensorflow as tf

from model_weights import age_gender_model
import const as c
OUTDIR = '../model_weights'
    
if __name__=='__main__':
    output_name = c.TFKERAS_MODEL_NAME
    age_gender_weight_name = c.AGE_GENDER_WEIGHT_NAME
    model = age_gender_model.load_model()
    model.load_weights(age_gender_weight_name)
    #model = tf.keras.models.load_model(age_gender_weight_name)
    print('tfkeras model saving....')
    model.save(os.path.join(OUTDIR, output_name))

