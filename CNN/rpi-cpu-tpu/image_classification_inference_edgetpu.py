import argparse
import time
import os
import numpy as np
from PIL import Image
from model import classify_tflite as classify
import tflite_runtime.interpreter as tflite
import platform


EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


def load_data(input_shape):
    load_data = []
    image_path = './dataset/imagenet/imagenet_1000_raw/'
    input_files = os.listdir(image_path)

    for image_file in input_files:
        image = Image.open(image_path+'/'+image_file)
        image = image.convert('RGB').resize([input_shape[1], input_shape[2]], Image.ANTIALIAS)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        load_data.append(image)

    return load_data


def inference(dataset, interpreter, input_index, top_k, threshold):
    iter_times, accuracy = [], []

    for image in dataset:
        interpreter.set_tensor(input_index, image)
        start = time.perf_counter() 
        interpreter.invoke()
        iter_times.append(time.perf_counter() - start)  

        classes = classify.get_output(interpreter, top_k, threshold) 
        for klass in classes:
            accuracy.append(klass.score)

    return accuracy, iter_times


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')

  args = parser.parse_args()
  
  global top_k, threshold

  top_k = args.top_k
  threshold = args.threshold

  model_load_time = time.time()
  interpreter = make_interpreter(args.model)
  input_details = interpreter.get_input_details()
  input_index = input_details[0]['index']
  input_shape = input_details[0]['shape']
  interpreter.allocate_tensors()
  model_load_time = time.time() - model_load_time

  dataset_load_time=time.time()
  dataset = load_data(input_shape)
  dataset_load_time = time.time() - dataset_load_time
    
  inference_time = time.time()
  accuracy, iter_times = inference(dataset, interpreter, input_index, top_k, threshold)
  inference_time = time.time() - inference_time

  print('***** TF-lite matric *****')
  print('accuracy = {:.3f}'.format(np.sum(accuracy)/(len(dataset)*len(dataset[0]))))
  print('model_load_time = {:.3f}'.format(model_load_time))
  print('dataset_load_time = {:.3f}'.format(dataset_load_time))
  print('inference_time = {:.3f}'.format(inference_time))
  print('inference_time(avg) = {:.3f}'.format(np.sum(iter_times) / (len(dataset)*len(dataset[0]))))
  print('IPS = {:.3f}'.format((len(dataset)*len(dataset[0])) / (model_load_time + dataset_load_time + inference_time)))
  print('IPS(inf) = {:.3f}'.format((len(dataset)*len(dataset[0])) / np.sum(iter_times)))
  


if __name__ == '__main__':
  main()
