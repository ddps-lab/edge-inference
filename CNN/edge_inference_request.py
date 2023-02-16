from flask import Flask
import tensorflow as tf
import numpy as np
import psutil
import shutil

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.18*1024)])

  print("1:", psutil.virtual_memory())

from tensorflow.keras.applications import (
        mobilenet,
        mobilenet_v2,
        inception_v3
        )

models = {
        'mobilenet':mobilenet,
        'mobilenet_v2':mobilenet_v2,
        'inception_v3':inception_v3
        }

models_detail = {
        'mobilenet':mobilenet.MobileNet(weights='imagenet'),
        'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet'),
        'inception_v3':inception_v3.InceptionV3(weights='imagenet')
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

# def mobilenetv1_image_preprocess(image_array):
#     return mobilenet.preprocess_input(
#         image_array[tf.newaxis, ...])
#
#
# def mobilenetv2_image_preprocess(image_array):
#     return mobilenet_v2.preprocess_input(
#         image_array[tf.newaxis, ...])
#
#
# def inceptionv3_image_preprocess(image_array):
#     return inception_v3.preprocess_input(
#         image_array[tf.newaxis, ...])

mobilenetv1_image_path = './dataset/imagenet/imagenet_1000_raw/n02782093_1.JPEG'
mobilenetv2_image_path = './dataset/imagenet/imagenet_1000_raw/n04404412_1.JPEG'
inceptionv3_image_path = './dataset/imagenet/imagenet_1000_raw/n13040303_1.JPEG'

mobilenetv1_test_image = mobilenet_load_image(mobilenetv1_image_path)
mobilenetv2_test_image = mobilenet_load_image(mobilenetv2_image_path)
inceptionv3_test_image = inception_load_image(inceptionv3_image_path)

mobilenetv1_test_image_array = image_to_array(mobilenetv1_test_image)
mobilenetv2_test_image_array = image_to_array(mobilenetv2_test_image)
inceptionv3_test_image_array = image_to_array(inceptionv3_test_image)

# mobilenetv1_test_image_preprocessed = mobilenetv1_image_preprocess(mobilenetv1_test_image_array)
# mobilenetv2_test_image_preprocessed = mobilenetv2_image_preprocess(mobilenetv2_test_image_array)
# inceptionv3_test_image_preprocessed = inceptionv3_image_preprocess(inceptionv3_test_image_array)

mobilenetv1_test_image_preprocessed = image_preprocess(mobilenetv1_test_image_array, 'mobilenet')
mobilenetv2_test_image_preprocessed = image_preprocess(mobilenetv2_test_image_array, 'mobilenet_v2')
inceptionv3_test_image_preprocessed = image_preprocess(inceptionv3_test_image_array, 'inception_v3')


def save_model(model, saved_model_dir):
    model = models_detail[model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    model.save(saved_model_dir, include_optimizer=False, save_format='tf')

loaded_models = {}

for model in models.keys():
    path = f'{model}_saved_model'
    save_model(model, path)
    loaded_models[model] = tf.keras.models.load_model(path)

app = Flask(__name__)

@app.route('/mobilenetv1')
def mobilenetv1():
    result = loaded_models['mobilenet'].predict(mobilenetv1_test_image_preprocessed)
    return result

app.run(host='localhost', port=5001)