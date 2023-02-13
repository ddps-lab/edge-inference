import numpy as np
import os
import time
import pathlib
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.3*1024)])

image_size = 224
batch_size = 128

model_load_start=time.time()
#mobilenetv1_model = tf.keras.applications.MobileNet(weights='imagenet')
mobilenetv2_model = tf.keras.applications.MobileNetV2(weights='imagenet')
#inceptionv3_model = tf.keras.applications.InceptionV3(weights='imagenet')
model_load_time=time.time()-model_load_start

image_path = pathlib.Path('./dataset/imagenet/imagenet_1000_raw')
raw_image_path = sorted(list(image_path.glob('*.JPEG')))

def mobilenetv1_image_preprocess(image_array):
    return tf.keras.applications.mobilenet.preprocess_input(
        image_array[tf.newaxis, ...])

def mobilenetv2_image_preprocess(image_array):
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array[tf.newaxis, ...])

def inceptionv3_image_preprocess(image_array):
    return tf.keras.applications.inception_v3.preprocess_input(
        image_array[tf.newaxis, ...])
    
data_load_time = []
raw_inference_time = []
inference_time = []
accuracy=[]

num_batches = (len(raw_image_path) + batch_size - 1) // batch_size
for i in range(num_batches):
    batch_files = raw_image_path[i * batch_size:(i + 1) * batch_size]
    for f in batch_files:
      data_load_start = time.time()
      images = tf.keras.preprocessing.image.load_img(f,target_size=[image_size, image_size])
      data_load_time.append(time.time() - data_load_start)
      test_image_array = tf.keras.preprocessing.image.img_to_array(images, dtype=np.int32)
      test_image_preprocessed = mobilenetv2_image_preprocess(test_image_array)


    iference_start = time.time()
    raw_inference_start = time.time()
    preds = mobilenetv2_model.predict(test_image_preprocessed)
    raw_inference_time.append(time.time() - raw_inference_start)
    accuracy.append(tf.keras.applications.imagenet_utils.decode_predictions(preds, top=1)[0][0][2])
    inference_time.append(time.time() - iference_start)


print("accuracy =", np.sum(accuracy) / len(accuracy))
print("model_load_time =", model_load_time)
print("data_load_time =", np.sum(data_load_time))
print("total_task_inference_time =", np.sum(inference_time))
print("inference_time(avg) =", np.sum(raw_inference_time) / 1000)
print("IPS =", 1000 / (model_load_time + np.sum(data_load_time) + np.sum(inference_time)))
print('IPS(inf) =', 1000 / np.sum(raw_inference_time))
