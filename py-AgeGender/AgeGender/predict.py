import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import cv2
import numpy as np
from models import age_gender_model
from age_gender_utils.identity_age import PostProcess, return_generation
from age_gender_utils.load import to_mean_pixel, MEAN_AVG


x = 100
y = 250
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_PLAIN,
               font_scale=1.5, thickness=2):
    text_color = (255, 255, 255)
    cv2.putText(image, label, point, font, font_scale, 
            text_color, thickness, lineType=cv2.LINE_AA)


def predict(img_path):
    age_gender_models = age_gender_model.load_model()
    age_gender_models.load_weights('weights/cp-age-gender_01.hdf5')
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    cimg = img.copy()
    img = to_mean_pixel(img, MEAN_AVG)
    img = img.astype(np.float32)/255
    img = np.reshape(img, (1, 299, 299, 3))

    generation, iden, gender = age_gender_models.predict(img)
    pred_identity = PostProcess(generation, iden).post_age_process()
    
    pred_gender = "Male" if np.argmax(gender) < 0.5 else "Female"
    print('gender is {0} and predict age is {1}'.format(pred_gender, pred_identity))
    label = "{0} age  {1}".format(int(pred_identity), "Male" if np.argmax(gender) < 0.5 else "Female")

    draw_label(cimg, (x, y), label)
    cv2.imwrite('prediction.png', cimg)

if __name__=='__main__':
    img_path = str(sys.argv[1])
    predict(img_path)


