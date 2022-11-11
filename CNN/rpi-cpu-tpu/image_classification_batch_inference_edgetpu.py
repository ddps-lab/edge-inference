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


def load_data(batch_size):
    load_data=[[] for _ in range(1000//batch_size)]
    image_path = './dataset/imagenet/imagenet_1000_raw/'
    input_files = os.listdir(image_path)

    for idx, image_file in enumerate(input_files):
        image = Image.open(image_path+'/'+image_file)
        image = image.convert('RGB').resize([299,299], Image.ANTIALIAS)
        image = np.array(image)
        
        if idx//batch_size < len(load_data):
            load_data[idx//batch_size].append(image)
        else:
            break

    return load_data


def inference(interpreter, args, batch_size, image_batch): 
    iter_times=[]
    accuracy=[]

    inference_start = time.time()

    for batch in image_batch: 
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], batch)
        start = time.perf_counter() 
        interpreter.invoke()
        iter_times.append(time.perf_counter() - start)   
        classes = classify.get_output(interpreter, args.top_k, args.threshold)

        for klass in classes:
            accuracy.append(klass.score)

        accuracy.sort(reverse=True)

    inference_time = time.time() - inference_start
    
    return accuracy, inference_time, iter_times


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
  parser.add_argument(
      '-b', '--batch_size', type=int, default=1,
      help='Size of input data batch')
  args = parser.parse_args()
  
  batch_size = args.batch_size

  model_load_time = time.time()
  interpreter = make_interpreter(args.model)
  tensor_index = interpreter.get_input_details()[0]['index']
  interpreter.resize_tensor_input(tensor_index, [batch_size, 299, 299, 3])
  interpreter.allocate_tensors()
  model_load_time = time.time() - model_load_time

  dataset_load_time=time.time()
  image_batch = load_data(batch_size)
  dataset_load_time = time.time() - dataset_load_time
    
  accuracy, inference_time, iter_times = inference(interpreter, args, batch_size, image_batch)

  print('***** TF-lite matric *****')
  print('accuracy =', np.sum(accuracy)/(len(image_batch)*len(image_batch[0])))
  print('model_load_time =', model_load_time)
  print('dataset_load_time =', dataset_load_time)
  print('inference_time =', inference_time)
  print('inference_time(avg) =', np.sum(iter_times) / (len(image_batch)*len(image_batch[0])))
  print('IPS =', (len(image_batch)*len(image_batch[0])) / (model_load_time + dataset_load_time + inference_time))
  print('IPS(inf) =', (len(image_batch)*len(image_batch[0])) / np.sum(iter_times))
  

if __name__ == '__main__':
  main()
