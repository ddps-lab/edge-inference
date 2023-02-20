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

img_size = 224

def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                   'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                   'image/class/text': tf.io.FixedLenFeature([], tf.string, '')}
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)
    label_text = tf.cast(obj['image/class/text'], tf.string)
    return imgdata, label, label_text


def val_preprocessing(record):
    imgdata, label, label_text = deserialize_image_record(record)
    label -= 1
    image = tf.io.decode_jpeg(imgdata, channels=3,
                              fancy_upscaling=False,
                              dct_method='INTEGER_FAST')

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    side = tf.cast(tf.convert_to_tensor(256, dtype=tf.int32), tf.float32)

    scale = tf.cond(tf.greater(height, width),
                    lambda: side / width,
                    lambda: side / height)

    new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
    new_width = tf.cast(tf.math.rint(width * scale), tf.int32)

    image = tf.image.resize(image, [new_height, new_width], method='bicubic')
    image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)

    image = tf.keras.applications.mobilenet.preprocess_input(image)

    return image, label, label_text


def get_dataset(batch_size, use_cache=False):
    data_dir = './dataset/imagenet/imagenet_1000'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)

    if use_cache:
        shutil.rmtree('tfdatacache', ignore_errors=True)
        os.mkdir('tfdatacache')
        dataset = dataset.cache(f'./tfdatacache/imagenet_val')

    return dataset


batchsize = 1

mobilenet_dataset = get_dataset(batchsize)
print(mobilenet_dataset)

img_size = 299
inception_dataset = get_dataset(batchsize)


def save_model(model, saved_model_dir):
    model = models_detail[model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    model.save(saved_model_dir, include_optimizer=False, save_format='tf')


loaded_models = {}

model_names = models_detail.keys()
for model_name in model_names:
    model_path = f'{model_name}_saved_model'
    if os.path.isdir(model_path) == False:
        print('model save')
        save_model(model_name, model_path)
    loaded_models[model_name] = tf.keras.models.load_model(model_path)

app = Flask(__name__)

mobilenetv1_model = loaded_models['mobilenet']


@app.route('/mobilenetv1')
def mobilenetv1():
    inference_start_time = time.time()
    for i, (validation_ds, batch_labels, _) in enumerate(mobilenet_dataset):
        # result = loaded_models['mobilenet'].predict(mobilenetv1_test_image_preprocessed)
        mobilenetv1_model(validation_ds)
        break
    inference_time = time.time() - inference_start_time

    return f'mobilenetv1 inference success\ntime:{inference_time}\n'


@app.route('/mobilenetv2')
def mobilenetv2():
    inference_start_time = time.time()
    for i, (validation_ds, batch_labels, _) in enumerate(mobilenet_dataset):
        # result = loaded_models['mobilenet_v2'].predict(mobilenetv2_test_image_preprocessed)
        model = loaded_models['mobilenet_v2']
        model(validation_ds)
        break
    inference_time = time.time() - inference_start_time
    return f'mobilenetv2 inference success\ntime:{inference_time}\n'


@app.route('/inceptionv3')
def inceptionv3():
    inference_start_time = time.time()
    for i, (validation_ds, batch_labels, _) in enumerate(mobilenet_dataset):
        # result = loaded_models['inception_v3'].predict(inceptionv3_test_image_preprocessed)
        model = loaded_models['inception_v3']
        model(validation_ds)
        break
    inference_time = time.time() - inference_start_time

    return f'inceptionv3 inference success\ntime:{inference_time}\n'


app.run(host='localhost', port=5001)
