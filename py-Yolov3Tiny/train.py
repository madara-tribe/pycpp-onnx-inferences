import argparse
import numpy as np
import os
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import *
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from callbacks.triangular3 import Triangular3Scheduler
from callbacks.lr_finder import LRFinder


NUM_RPOCH = 100
batch_size = 20
DARKNET_WEIGHT_PATH = 'weight/yolo.h5'


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--annotation_path", type=str, default='../../../anchors_classes/yolov3_voc.txt',
                        help="annotation path")
    parser.add_argument("--classes_path", type=str, default='../../../anchors_classes/classes.txt',
                        help='classes path')
    parser.add_argument("--anchors_path", type=str, default='../../../anchors_classes/tiny_yolo_anchors.txt',
                        help='anchors path')
    parser.add_argument("--lod_dir", type=str, default='../../../logs',
                        help='lod dir')
    return parser

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=None, freeze_body=2,
            weights_path=DARKNET_WEIGHT_PATH):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=None, freeze_body=2,
            weights_path=None):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def train():
    opts = get_argparser().parse_args()
    log_dir = opts.lod_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    class_names = get_classes(opts.classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(opts.anchors_path)

    input_shape = (416,416) # multiple of 32, hw
    is_tiny_version = len(anchors)==6 # default setting
    print(is_tiny_version)

    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=None)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=None) # make sure you know what you freeze
    model.summary()

    val_split = 0.1
    with open(opts.annotation_path) as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, val on {} samples'.format(num_train, num_val))

    print('freeze all of the layers at 1nd train')
    #model.load_weights('drive/My Drive/ep029-loss19.371-val_loss18.962.h5')
    model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    checkpoint = ModelCheckpoint(log_dir+'/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    callback = [reduce_lr, checkpoint]

    lr_callback  = None
    if lr_callback:
        TOTAL_ITERATIONS = 10000
        MIN_LR = 1e-9
        MAX_LR = 3e-4
        lr_finder = LRFinder(min_lr=1e-10, max_lr=2e-2, steps_per_epoch=TOTAL_ITERATIONS, epochs=1)
        loss_schedule = Triangular3Scheduler(min_lr=MIN_LR, max_lr=MAX_LR,
                                            steps_per_epoch=np.ceil(num_train/batch_size), lr_decay=1.0,
                                            cycle_length=1, mult_factor=0.99)
        callback = [reduce_lr, checkpoint, loss_schedule]

    try:
      model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=NUM_RPOCH, initial_epoch=0, callbacks = callback)
    finally:
      model.save_weights(log_dir + '/yolov3_weights_stage_1.h5')

if __name__ == '__main__':
    train()

