import os
import time
import numpy as np
import shutil
import requests
from functools import partial
import argparse
import tensorflow as tf
import psutil
import tflite_runtime.interpreter as tflite


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.3*1024)])


results = None
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', default=1, type=int)
parser.add_argument('--case', type=str, required=True)
parser.add_argument('--engines',default=1, type=int)
parser.add_argument('--img_size',default=224, type=int)
args = parser.parse_args()
batch_size = args.batchsize
case=args.case
num_engines=args.engines
img_size=args.img_size


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
    data_dir = './imagenet_1000'
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
    

def calibrate_fn(n_calib, batch_size, dataset):
    for i, (calib_image, _, _) in enumerate(dataset):
        if i > n_calib // batch_size:
            break
        yield (calib_image,)


def predict_tflite(batch_size):
   
    model_load_time = time.time()
    model = tflite.Interpreter("./yolov5s-fp16_edgetpu.tflite", experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    model.allocate_tensors()
    model_load_time = time.time() - model_load_time


    display_every = 5000
    display_threshold = display_every
    
    pred_labels = []
    actual_labels = []
    iter_times = []
    
    dataset_load_time = time.time()
    dataset = get_dataset(batch_size)
    dataset_load_time = time.time() - dataset_load_time

    input_details = model.get_input_details()[0]['shape']
    output_details = model.get_output_details()[0]['shape']

    
    iftime_start = time.time()
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        input_data = np.array(np.random.random_sample(validation_ds), dtype=np.float32)
        start_time = time.time()
        model.set_tensor(input_details[0]['shape'], input_data)
        model.invoke()
        pred_prob_keras=model.get_tensor(output_details)
        iter_times.append(time.time() - start_time)

        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        pred_labels.extend(list(np.argmax(pred_prob_keras, axis=1)))
        
        if i*batch_size >= display_threshold:
            display_threshold+=display_every
            
    iter_times = np.array(iter_times)
    acc_keras_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    
    
    print('***** TF-lite matric *****')
    print('user_batch_size =', batch_size)
    print('accuracy =', acc_keras_gpu)
    print('model_load_time =', model_load_time)
    print('dataset_load_time =', dataset_load_time)
    print('inference_time =', time.time() - iftime_start)
    print('inference_time(avg) =', np.sum(iter_times) / (len(iter_times)*batch_size))
    print('IPS =', (len(iter_times)*batch_size) / (model_load_time + dataset_load_time + (time.time() - iftime_start)))
    print('IPS(inf) =', (len(iter_times)*batch_size) / np.sum(iter_times))



if case == 'tflite' :
    predict_tflite(batch_size)
