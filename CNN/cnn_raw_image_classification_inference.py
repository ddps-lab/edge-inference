import numpy as np
import pathlib
import os
import time
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.3*1024)])
  

model_load_start=time.time()
#mobilenetv1_model = tf.keras.applications.MobileNet(weights='imagenet')
#mobilenetv2_model = tf.keras.applications.MobileNetV2(weights='imagenet')
inceptionv3_model = tf.keras.applications.InceptionV3(weights='imagenet')
model_load_time=time.time()-model_load_start

image_path = pathlib.Path('./dataset/imagenet/imagenet_1000_raw')
raw_image_path = sorted(list(image_path.glob('*.JPEG')))

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

test_images_preprocessed = []
accuracy = []
data_load_time = []
raw_inference_time = []

for image_path in raw_image_path:
  data_load_start = time.time()
  test_image = inception_load_image(image_path)
  data_load_time.append(time.time() - data_load_start)
  test_image_array = image_to_array(test_image)
  test_image_preprocessed = inceptionv3_image_preprocess(test_image_array)
  test_images_preprocessed.append(test_image_preprocessed)

total_iference_start = time.time()
for inference_image in test_images_preprocessed:
  raw_inference_start = time.time()
  preds = inceptionv3_model.predict(inference_image)
  raw_inference_time.append(time.time() - raw_inference_start)
  accuracy.append(tf.keras.applications.imagenet_utils.decode_predictions(preds, top=1)[0][0][2])
total_task_inference_time = time.time() - total_iference_start


print("accuracy =", np.sum(accuracy) / 1000)
print("model_load_time =", model_load_time)
print("data_load_time =", np.sum(data_load_time))
print("total_task_inference_time =", total_task_inference_time)
print("inference_time(avg) =", np.sum(raw_inference_time) / 1000)
print("IPS =", 1000 / (model_load_time + np.sum(data_load_time) + total_task_inference_time))
print('IPS(inf) =', 1000 / np.sum(raw_inference_time))
