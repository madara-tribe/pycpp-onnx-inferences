import os, sys
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('../models')
import time
import cv2
import numpy as np

from yolov3_utils.utils import get_base_parser, update_parser, get_savepath
from yolov3_utils import webcamera as webcamera_utils
from yolov3_utils.detector_utils import plot_results, load_image
from models import age_model, age_gender_onnx_predict, gender_model
from yolov3_utils import identity_age, load
from models.onnx_inference import age_onnx_inference, gender_onnx_inference

from logging import getLogger
logger = getLogger(__name__)


yolov3_model_existed_is = None
if yolov3_model_existed_is:
    from models import yolov3_face_detection

ONNX_PREDICT = True


WEIGHT_PATH = 'yolov3-face.onnx'
AGE_GENDER_ONNX_PATH = '../models/age_gender_model.onnx'
IMAGE_PATH = 'input_person.jpg'
SAVE_IMAGE_PATH = 'output.png'
AGE_GENDER_SAVE_NAME = 'result_output.jpg'

IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
FACE_CATEGORY = ['face']
THRESHOLD = 0.2
IOU = 0.45


parser = get_base_parser(
    'Yolov3 face detection model', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)
age_model = age_model.load_model()
age_model.load_weights('../weights/cp-age-gender_06.hdf5')

gender_models = gender_model.load_model()
gender_models.load_weights('../weights/gender_bestep15.hdf5')


def age_gender_postprocess(crop_faces, crop_pos, cimg):
    print('predicting age and gender .....')
    if ONNX_PREDICT:
        gender = gender_onnx_inference(crop_faces, GENDER_ONNX_PATH)
        generation, idens, _ = age_onnx_inference(crop_faces, AGE_ONNX_PATH)
    else:
        gender = gender_models.predict(crop_faces)
        generation, idens, _ = age_model.predict(crop_faces)
        
    pred_identity_ages = np.array([identity_age.PostProcess(gene, iden).post_age_process() for gene, iden in zip(generation, idens)])
    pred_identity_ages = pred_identity_ages.reshape(len(pred_identity_ages), 1)
    predicted_genders = np.array([np.argmax(gen) for gen in gender])
    
    for i, d in enumerate(crop_pos):
        label = "{0}{1}".format(int(pred_identity_ages[i]),
                                    "M" if predicted_genders[i] < 0.5 else "F")
        draw_label(cimg, (d[0], d[1]), label)
    cv2.imwrite(AGE_GENDER_SAVE_PATH, cimg)
    

def detection_from_image():
    # net initialize
    detector = ailia.Detector(
        MODEL_PATH,
        WEIGHT_PATH,
        len(FACE_CATEGORY),
        format=ailia.NETWORK_IMAGE_FORMAT_RGB,
        channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST,
        range=ailia.NETWORK_IMAGE_RANGE_U_FP32,
        algorithm=ailia.DETECTOR_ALGORITHM_YOLOV3,
        env_id=args.env_id
    )

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        img = load_image(image_path)
        logger.debug(f'input image shape: {img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                detector.compute(img, THRESHOLD, IOU)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            detector.compute(img, THRESHOLD, IOU)

         # plot result
        cimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGRA2BGR)
        res_img, crop_faces, crop_pos = plot_results(detector, img, FACE_CATEGORY)
        
        print('predicting age and gender .....')
        age_gender_postprocess(crop_faces, crop_pos, cimg)
        
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)
    logger.info('Script finished successfully.')
    
    
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_PLAIN,
               font_scale=1.2, thickness=1):
    text_color = (255, 255, 255)
    cv2.putText(image, label, point, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def detection_from_video():
    # yolov3 net
    detector = yolov3_face_detection()

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        writer = webcamera_utils.get_writer(
            args.savepath, IMAGE_HEIGHT, IMAGE_WIDTH
        )
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        _, resized_img = webcamera_utils.adjust_frame_size(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH
        )

        img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2BGRA)
        detector.compute(img, THRESHOLD, IOU)
        
        cimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGRA2BGR)
        res_img, crop_faces, crop_pos = plot_results(detector, resized_img, FACE_CATEGORY, False)
        
        age_gender_postprocess(crop_faces, crop_pos, cimg)
        cv2.imshow('frame', res_img)

        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')
    


if __name__ == '__main__':
    video_mode = None
    if video_mode:
        detection_from_video()
    else:
        detection_from_image()
