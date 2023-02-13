import numpy as np
import pathlib
import os
import time
import tensorflow as tf
import requests
import json
import roundrobin
from numpy import random
from threading import Thread


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


def mobilenetv1_image_preprocess(image_array):
    return tf.keras.applications.mobilenet.preprocess_input(
        image_array[tf.newaxis, ...])


def mobilenetv2_image_preprocess(image_array):
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array[tf.newaxis, ...])


def inceptionv3_image_preprocess(image_array):
    return tf.keras.applications.inception_v3.preprocess_input(
        image_array[tf.newaxis, ...])


# Download human-readable labels for ImageNet.
imagenet_labels_url = (
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
)
response = requests.get(imagenet_labels_url)
# Skiping backgroung class
labels = [x for x in response.text.split("\n") if x != ""][1:]
# Convert the labels to the TensorFlow data format
tf_labels = tf.constant(labels, dtype=tf.string)


def postprocess(prediction, labels=tf_labels):
    """Convert from probs to labels."""
    indices = tf.argmax(prediction, axis=-1)  # Index with highest prediction
    label = tf.gather(params=labels, indices=indices)  # Class name
    return label


mobilenetv1_image_path = './dataset/imagenet/imagenet_1000_raw/n02782093_1.JPEG'
mobilenetv2_image_path = './dataset/imagenet/imagenet_1000_raw/n04404412_1.JPEG'
inceptionv3_image_path = './dataset/imagenet/imagenet_1000_raw/n13040303_1.JPEG'

mobilenetv1_test_image = mobilenet_load_image(mobilenetv1_image_path)
mobilenetv2_test_image = mobilenet_load_image(mobilenetv2_image_path)
inceptionv3_test_image = inception_load_image(inceptionv3_image_path)

mobilenetv1_test_image_array = image_to_array(mobilenetv1_test_image)
mobilenetv2_test_image_array = image_to_array(mobilenetv2_test_image)
inceptionv3_test_image_array = image_to_array(inceptionv3_test_image)

mobilenetv1_test_image_preprocessed = mobilenetv1_image_preprocess(mobilenetv1_test_image_array)
mobilenetv2_test_image_preprocessed = mobilenetv2_image_preprocess(mobilenetv2_test_image_array)
inceptionv3_test_image_preprocessed = inceptionv3_image_preprocess(inceptionv3_test_image_array)

SERVER_URL = "http://172.17.0.2:8501/v1/models/"
headers = {"content-type": "application/json"}
MAX = 10
models = [('mobilenetv1', 1), ('mobilenetv2', 3), ('inceptionv3', 6)]
datas = {
    'mobilenetv1': json.dumps(
        {"signature_name": "serving_default", "instances": mobilenetv1_test_image_preprocessed.tolist()}),
    'mobilenetv2': json.dumps(
        {"signature_name": "serving_default", "instances": mobilenetv2_test_image_preprocessed.tolist()}),
    'inceptionv3': json.dumps(
        {"signature_name": "serving_default", "instances": inceptionv3_test_image_preprocessed.tolist()})
}

time_tracking = {}


def ModelRequest(model, data):
    inference_start = time.time()
    url = SERVER_URL + model + ':predict'
    res = requests.post(url, data, headers)
    response = json.loads(res.text)['predictions']
    end_time = time.time() - inference_start
    key = model + " " + str(inference_start)
    time_tracking[key] = end_time

    return response


get_weighted_smooth = roundrobin.smooth(models)
model_sequence = [get_weighted_smooth() for _ in range(MAX)]
RoundPerEvent = 10
TotalEvent = 10
poisson_distribution = random.poisson(RoundPerEvent, TotalEvent)

if __name__ == "__main__":
    request_start = time.time()
    threads = []

    for events in poisson_distribution:
        print('request', events)
        event_start = time.time()

        for model_idx in range(events):
            model = model_sequence[model_idx % len(model_sequence)]
            th = Thread(target=ModelRequest, args=(model, datas[model]))
            th.start()
            threads.append(th)

            # case : linear
            # ModelRequest(model, datas[model])

    for thread in threads:
        thread.join()
    request_end = time.time() - request_start

    print("Return value:", poisson_distribution)
    print("Length of return value:", len(poisson_distribution))
    print("total request time", request_end)

    print(time_tracking)
    with open(str(time.time()) + '.txt', 'w') as file:
        file.write(json.dumps(time_tracking))
