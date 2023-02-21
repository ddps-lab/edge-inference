import argparse

from flask import Flask
import tensorflow as tf
import numpy as np
import shutil
import os
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(
            memory_limit=0.6 * 1024)])

from tensorflow.keras.applications import (
    mobilenet,
    mobilenet_v2,
    inception_v3
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='all', type=str)
args = parser.parse_args()
models_to_load = args.model

models = {
    'mobilenet': mobilenet,
    'mobilenet_v2': mobilenet_v2,
    'inception_v3': inception_v3
}

models_detail = {
    'mobilenet': mobilenet.MobileNet(weights='imagenet'),
    'mobilenet_v2': mobilenet_v2.MobileNetV2(weights='imagenet'),
    'inception_v3': inception_v3.InceptionV3(weights='imagenet')
}


def mobilenet_load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[224, 224])


def inception_load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[299, 299])


def image_to_array(image):
    return tf.keras.preprocessing.image.img_to_array(image, dtype=np.int32)


def image_preprocess(image_array, model_name):
    return models[model_name].preprocess_input(
        image_array[tf.newaxis, ...])


mobilenetv1_image_path = './dataset/imagenet/imagenet_1000_raw/n02782093_1.JPEG'
mobilenetv2_image_path = './dataset/imagenet/imagenet_1000_raw/n04404412_1.JPEG'
inceptionv3_image_path = './dataset/imagenet/imagenet_1000_raw/n13040303_1.JPEG'

mobilenetv1_test_image = mobilenet_load_image(mobilenetv1_image_path)
mobilenetv2_test_image = mobilenet_load_image(mobilenetv2_image_path)
inceptionv3_test_image = inception_load_image(inceptionv3_image_path)

mobilenetv1_test_image_array = image_to_array(mobilenetv1_test_image)
mobilenetv2_test_image_array = image_to_array(mobilenetv2_test_image)
inceptionv3_test_image_array = image_to_array(inceptionv3_test_image)

mobilenetv1_test_image_preprocessed = image_preprocess(mobilenetv1_test_image_array, 'mobilenet')
mobilenetv2_test_image_preprocessed = image_preprocess(mobilenetv2_test_image_array, 'mobilenet_v2')
inceptionv3_test_image_preprocessed = image_preprocess(inceptionv3_test_image_array, 'inception_v3')


def save_model(model, saved_model_dir):
    model = models_detail[model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    model.save(saved_model_dir, include_optimizer=False, save_format='tf')


loaded_models = {}

if models_to_load == 'all':
    model_names = models_detail.keys()
    for model_name in model_names:
        model_path = f'{model_name}_saved_model'
        if os.path.isdir(model_path) == False:
            print('model save')
            save_model(model_name, model_path)
        loaded_models[model_name] = tf.keras.models.load_model(model_path)
elif models_to_load in models_detail.keys():
    model_path = f'{models_to_load}_saved_model'
    if os.path.isdir(model_path) == False:
        print('model save')
        save_model(models_to_load, model_path)
    loaded_models[models_to_load] = tf.keras.models.load_model(model_path)


app = Flask(__name__)


@app.route('/mobilenetv1')
def mobilenetv1():
    inference_start_time = time.time()
    result = loaded_models['mobilenet'].predict(mobilenetv1_test_image_preprocessed)
    inference_time = time.time() - inference_start_time

    print(result)
    
    return f'mobilenetv1 inference success\ntime:{inference_time}\n'


@app.route('/mobilenetv2')
def mobilenetv2():
    inference_start_time = time.time()
    result = loaded_models['mobilenet_v2'].predict(mobilenetv2_test_image_preprocessed)
    inference_time = time.time() - inference_start_time

    print(result)

    return f'mobilenetv2 inference success\ntime:{inference_time}\n'


@app.route('/inceptionv3')
def inceptionv3():
    inference_start_time = time.time()
    result = loaded_models['inception_v3'].predict(inceptionv3_test_image_preprocessed)
    inference_time = time.time() - inference_start_time

    print(result)

    return f'inceptionv3 inference success\ntime:{inference_time}\n'


app.run(host='localhost', port=5001)
