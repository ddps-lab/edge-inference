import os
import time
import numpy as np
import shutil
import requests
from functools import partial
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import psutil

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

print("2:", psutil.virtual_memory())


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


results = None
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', default=1, type=int)
parser.add_argument('--model', default='mobilenet', type=str)
parser.add_argument('--case', type=str, required=True)
parser.add_argument('--quantization',default='FP32',type=str)
parser.add_argument('--engines',default=1, type=int)
parser.add_argument('--img_size',default=224, type=int)
args = parser.parse_args()
batch_size = args.batchsize
model = args.model
case=args.case
quantization = args.quantization
num_engines=args.engines
img_size=args.img_size


def load_save_model(model, saved_model_dir = 'mobilenet_saved_model'):
    model = models_detail[model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    model.save(saved_model_dir, include_optimizer=False, save_format='tf')


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
    data_dir = './validation-00000-of-00001'
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



def build_FP_tensorrt_engine(model, quantization, batch_size):

    dataset = get_dataset(batch_size)

    if quantization == 'FP32':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.FP32,
                                                        maximum_cached_engines=num_engines,
                                                        max_workspace_size_bytes=8000000000)
    elif quantization == 'FP16':                                                 
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.FP16,
                                                        maximum_cached_engines=num_engines,
                                                        max_workspace_size_bytes=8000000000)
    
    elif quantization == 'INT8':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.INT8, 
                                                        max_workspace_size_bytes=8000000000, 
                                                        maximum_cached_engines=num_engines,
                                                        use_calibration=True)

    
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=f'{model}_saved_model',
                                        conversion_params=conversion_params)
    
    if quantization=='INT8':
        n_calib=50
        converter.convert(calibration_input_fn=partial(calibrate_fn, n_calib, batch_size, dataset.shuffle(buffer_size=n_calib, reshuffle_each_iteration=True)))

    else:
        converter.convert()
        
    trt_compiled_model_dir = f'{model}_saved_models_{quantization}'
    converter.save(output_saved_model_dir=trt_compiled_model_dir)

    return trt_compiled_model_dir


def predict_tf(batch_size,saved_model_dir):
   
    print("3:", psutil.virtual_memory())
    model_load_time = time.time()
    model = tf.keras.models.load_model(saved_model_dir)
    model_load_time = time.time() - model_load_time
    print("4:", psutil.virtual_memory())


    display_every = 5000
    display_threshold = display_every
    
    pred_labels = []
    actual_labels = []
    iter_times = []
    
    print("5:", psutil.virtual_memory())
    dataset_load_time = time.time()
    dataset = get_dataset(batch_size)
    dataset_load_time = time.time() - dataset_load_time
    print("6:", psutil.virtual_memory())


    iftime_start = time.time()
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        start_time = time.time()
        pred_prob_keras = model(validation_ds)
        iter_times.append(time.time() - start_time)

        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        pred_labels.extend(list(np.argmax(pred_prob_keras, axis=1)))
        
        if i*batch_size >= display_threshold:
            display_threshold+=display_every
            
    iter_times = np.array(iter_times)
    acc_keras_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    
    print("7:", psutil.virtual_memory())
    
    print('***** TF-FP32 matric *****')
    print('user_batch_size =', batch_size)
    print('accuracy =', acc_keras_gpu)
    print('model_load_time =', model_load_time)
    print('dataset_load_time =', dataset_load_time)
    print('inference_time =', time.time() - iftime_start)
    print('inference_time(avg) =', np.sum(iter_times)/len(iter_times))
    print('IPS =', len(iter_times) / (model_load_time + dataset_load_time + (time.time() - iftime_start)))
    print('IPS(inf) =', len(iter_times) / np.sum(iter_times))


def predict_trt(trt_compiled_model_dir, quantization, batch_size):

    model_load_time = time.time()
    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures['serving_default']
    model_load_time = time.time() - model_load_time

    display_every = 5000
    display_threshold = display_every

    pred_labels = []
    actual_labels = []
    iter_times = []

    dataset_load_time = time.time()
    dataset = get_dataset(batch_size)
    dataset_load_time = time.time() - dataset_load_time

    iftime_start = time.time()
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        start_time = time.time()
        trt_results = model_trt(validation_ds)
        iter_times.append(time.time() - start_time)

        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        pred_labels.extend(list(tf.argmax(trt_results['predictions'], axis=1).numpy()))

        if i*batch_size >= display_threshold:
            display_threshold+=display_every

    iter_times = np.array(iter_times)
    acc_keras_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)


    print('***** TRT-quantization matric *****')
    print('user_batch_size =', batch_size)
    print('accuracy =', acc_keras_gpu)
    print('model_load_time =', model_load_time)
    print('dataset_load_time =', dataset_load_time)
    print('inference_time =', time.time() - iftime_start)
    print('inference_time(avg) =', np.sum(iter_times)/len(iter_times))
    print('IPS =', len(iter_times) / (model_load_time + dataset_load_time + (time.time() - iftime_start)))
    print('IPS(inf) =', len(iter_times) / np.sum(iter_times))


saved_model_dir = f'{model}_saved_model'
if model :
    load_save_model(model, saved_model_dir)

if case == 'tf' :
    predict_tf(batch_size,saved_model_dir)

elif case == 'trt' :
    trt_compiled_model_dir = build_FP_tensorrt_engine(model, quantization, batch_size)
    predict_trt(trt_compiled_model_dir, quantization, batch_size)
