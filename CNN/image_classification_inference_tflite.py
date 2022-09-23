import argparse
import time
import os
import numpy as np

from PIL import Image

from model import classify
import tflite_runtime.interpreter as tflite
import platform
import tensorflow as tf


def load_labels(path, encoding='utf-8'): # 현재 쓰이지 않음
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def load_data(batch_size):
    loaded_data=[]
    image_path = './dataset/imagenet/imagenet_1000_raw/'
    input_files = os.listdir(image_path)

    for image_file in input_files:
        image = tf.io.read_file(image_path+'/'+ image_file)
        tensor_img = tf.io.decode_image(image, channels=3)
        tensor_img = tf.image.convert_image_dtype(tensor_img, tf.uint8) 
        tensor_img = tf.image.resize(tensor_img, [299,299], antialias=True) 
        loaded_data.append(tensor_img)
    loaded_data =  (tf.data.Dataset.from_tensor_slices(loaded_data).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
#     loaded_data =  (tf.data.Dataset.from_tensor_slices(loaded_data).batch(batch_size, drop_remainder=True))
#     loaded_data = tf.data.Dataset.from_tensor_slices(loaded_data).padded_batch(batch_size) 

    return loaded_data


def inference(interpreter, top_k, threshold, batch_size, image_batch):  
    iter_times=[]
    accuracy=[]

    iftime_start = time.time()

    for i, batch in enumerate(image_batch): 

        batch = tf.cast(batch, tf.uint8) 
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], batch)
        start = time.perf_counter() 
        interpreter.invoke()
        iter_times.append(time.perf_counter() - start)       
        classes = classify.get_output(interpreter, top_k, threshold)
        for klass in classes:
            accuracy.append(klass.score)
            
    inference_time = time.time() - iftime_start
    
    return accuracy, inference_time, iter_times


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-b', '--batch_size', type=int, default=1,
      help='Size of input data batch')
  args = parser.parse_args()
  
  batch_size = args.batch_size
   
  # load model
  model_load_time = time.time()

  interpreter = tflite.Interpreter(args.model)
  tensor_index = interpreter.get_input_details()[0]['index']
  interpreter.resize_tensor_input(tensor_index, [batch_size, 299, 299, 3]) 
  interpreter.allocate_tensors()

  model_load_time = time.time() - model_load_time
    
  # load dataset
  dataset_load_time=time.time()

  image_batch = load_data(batch_size)
    
  dataset_load_time = time.time() - dataset_load_time
    
  # inference
  accuracy, inference_time, iter_times = inference(interpreter, args.top_k, args.threshold, batch_size, image_batch)
   
  print('***** TF-lite matric *****')
  print('accuracy =', np.sum(accuracy)/1000)
  print('model_load_time =', model_load_time)
  print('dataset_load_time =', dataset_load_time)
  print('inference_time =', inference_time)
  print('inference_time(avg) =', np.sum(iter_times) / 1000)
  print('IPS =', 1000 / (model_load_time + dataset_load_time + inference_time))
  print('IPS(inf) =', 1000 / np.sum(iter_times))


if __name__ == '__main__':
  main()
