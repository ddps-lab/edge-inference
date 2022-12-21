import tensorflow as tf
import numpy as np
import pathlib
import time


model_load_start = time.time()
model = tf.keras.applications.MobileNetV2()
model_load_time = time.time() - model_load_start

labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', labels_url)
labels = np.array(open(labels_path).read().splitlines())[1:]

image_path = pathlib.Path('./dataset/imagenet/imagenet_1000_raw')
raw_image_path = sorted(list(image_path.glob('*.JPEG')))

def load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[224, 224])

def image_to_array(image):
    return tf.keras.preprocessing.image.img_to_array(image, dtype=np.int32)

def image_preprocess(image_array):
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array[tf.newaxis, ...])

test_images = []
data_load_start = time.time()
for image_path in raw_image_path:
    test_image = load_image(image_path)
    test_image_array = image_to_array(test_image)
    test_images.append(test_image_array)
data_load_time = time.time() - data_load_start

test_images_preprocessed = []
for test_image in test_images:
    test_image_preprocessed = image_preprocess(test_image)
    test_images_preprocessed.append(test_image_preprocessed)

def get_tags(probs, labels, max_classes = 1, prob_threshold = 0.01):
    probs_mask = probs > prob_threshold
    probs_filtered = probs[probs_mask] * 100
    labels_filtered = labels[probs_mask]
    
    sorted_index = np.flip(np.argsort(probs_filtered))
    labels_filtered = labels_filtered[sorted_index][:max_classes]
    probs_filtered = probs_filtered[sorted_index][:max_classes].astype(np.float)
    return probs_filtered

raw_inference_time = []
accuracy = []
start = time.time()
for image_index in range(0, len(test_images)):
    test_image = test_images[image_index]
    test_image_preprocessed = test_images_preprocessed[image_index]
    start1 = time.time()
    probabilities = model(test_image_preprocessed)
    raw_inference_time.append(time.time() - start1)
    accuracy.append(get_tags(probabilities.numpy()[0], labels))
total_task_inference_time = time.time() - start

print("accuracy =", np.sum(accuracy) / 1000)
print("model_load_time =", model_load_time)
print("data_load_time =", data_load_time)
print("total_task_inference_time =", total_task_inference_time)
print("inference_time(avg) =", np.sum(raw_inference_time) / 1000)
print("IPS =", 1000 / (model_load_time + data_load_time + total_task_inference_time))
print('IPS(inf) =', 1000 / np.sum(raw_inference_time))

