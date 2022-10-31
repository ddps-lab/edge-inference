import argparse
import time
import os
import numpy as np
from PIL import Image
from model import classify_tflite as classify
import tflite_runtime.interpreter as tflite
import platform
from multiprocessing import Pool
from multiprocessing import Manager
import os


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


def init_interpreter(model_path, edgetpu_model_path, num_cpu, num_processes):
    global interpreter, input_index, is_tpu
    
    num_cpu.append(1)
    # Make EdgeTPU interpreter
    if len(num_cpu) == num_processes-1:
        load_time = time.time()
        interpreter = make_interpreter(edgetpu_model_path)
        interpreter.allocate_tensors()
        load_time = time.time() - load_time
        model_load_time.value = load_time
        is_tpu = 1
    # Make CPU interpreter
    else:
        interpreter = tflite.Interpreter(model_path)
        interpreter.allocate_tensors()
        is_tpu = 0

    input_index = interpreter.get_input_details()[0]['index']

def load_data(input_shape):
    load_data = []
    image_path = './dataset/imagenet/imagenet_1000_raw/'
    input_files = os.listdir(image_path)

    for image_file in input_files:
        image = Image.open(image_path+'/'+image_file)
        image = image.convert('RGB').resize([input_shape[1], input_shape[2]], Image.ANTIALIAS)
        image = np.array(image, dtype=np.uint8)
        image = np.expand_dims(image, axis=0)
        load_data.append(image)

    return load_data


def inference(image):
    
    inference_time = time.perf_counter()
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()

    classes = classify.get_output(interpreter, top_k, threshold)
    
    for klass in classes:
        accuracy.append(klass.score)

    if is_tpu == 1:
        tpu_iter_times.append(time.perf_counter() - inference_time)
    else:
        iter_times.append(time.perf_counter() - inference_time)

    return iter_times, accuracy, tpu_iter_times


def main():
  total_time = time.time()

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-a', '--edgetpu_model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-p', '--process', type=int, default=4)
  args = parser.parse_args()

  global top_k, threshold, iter_times, accuracy, model_load_time, tpu_iter_times

  top_k = args.top_k
  threshold = args.threshold
  num_processes = args.process

  # Get input shape
  interpreter = tflite.Interpreter(args.model)
  input_shape = interpreter.get_input_details()[0]['shape']
  del interpreter

  # Get dataset
  dataset_load_time=time.time()
  dataset = load_data(input_shape)
  dataset_load_time = time.time() - dataset_load_time

  # Create sharing variables
  manager = Manager()
  accuracy = manager.list()
  iter_times = manager.list()
  tpu_iter_times = manager.list()
  num_cpu = manager.list()
  model_load_time = manager.Value(float, 0.0)

  # Multiprocess inference
  inference_time = time.time()
  with Pool(processes=num_processes, initializer=init_interpreter, initargs=(args.model, args.edgetpu_model, num_cpu, num_processes, )) as p:
      result = p.map(inference, dataset, chunksize=1)
  inference_time = time.time() - inference_time
  
  total_time = time.time() - total_time

  iter_times = result[-1][0]
  accuracy = result[-1][1]
  tpu_iter_times = result[-1][2]
  model_load_time = model_load_time.value
  
  print(args.model)
  print('***** TF-lite matric *****')
  print('accuracy = {:.3f}'.format(np.sum(accuracy)/(len(dataset)*len(dataset[0]))))
  print('model_load_time = {:.3f}'.format(model_load_time))
  print('dataset_load_time = {:.3f}'.format(dataset_load_time))
  print('inference_time = {:.3f}'.format(inference_time-model_load_time))
  print('inference_time(avg) = {:.3f}'.format((inference_time-model_load_time) / (len(dataset)*len(dataset[0]))))
  print('invoke_time(avg) = {:.3f}'.format(np.sum(iter_times) / (len(dataset)*len(dataset[0]))))
  print('IPS = {:.3f}'.format((len(dataset)*len(dataset[0])) / total_time))
  print('IPS(inf) = {:.3f}'.format((len(dataset)*len(dataset[0])) / inference_time))
  print('total_time = {:.3f}'.format(total_time))

if __name__ == '__main__':
  main()
