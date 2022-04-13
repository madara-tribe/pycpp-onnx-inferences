from pathlib import Path
from tqdm import tqdm
import cv2, os
import numpy as np
from time import time
from .identity_age import IdentityAge, return_generation

output_path = '../age_gender.mat'
def load_mat(path):
    return scipy.io.loadmat(path)

ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
MEAN_AVG = float(130.509485819935)

def normalize_contrast(img, avg=MEAN_AVG):
    normailize_img = (img - np.mean(img))/np.std(img)
    return (normailize_img*avg)+avg


def to_mean_pixel(img, avg):
    return (img - 128)*(128/avg)
    

def load_age_gender_data(path="../../UTKFace", img_size=224, save_path=None):
    image_dir = Path(path)
    out_genders = []
    generation, identity_ages = [], []
    out_imgs = []
    for i, image_path in enumerate(tqdm(image_dir.glob("*.jpg"))):
        image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
        age, gender = image_name.split("_")[:2]
        if int(gender)>1:
            continue
        out_genders.append(int(gender))
        generation.append(return_generation(int(age)))
        identity_ages.append(IdentityAge(int(age)).identity_age())
        img = cv2.imread(str(image_path))
        #plt.imshow(img),plt.show()
        img = cv2.resize(img, (img_size, img_size))
        out_imgs.append(to_mean_pixel(img, MEAN_AVG))
        if i==1000:
            break
        if save_path:
            output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
          "db": "utk", "img_size": img_size, "min_score": -1}
            scipy.io.savemat(output_path, output)
    return out_imgs, out_genders, generation, identity_ages





