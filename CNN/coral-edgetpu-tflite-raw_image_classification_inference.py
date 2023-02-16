import argparse
import time
import os
import numpy as np

from PIL import Image

from model import classify
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])



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
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}

  model_load_time = time.time()
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()
  model_load_time = time.time() - model_load_time

  dataset_load_time=time.time()
  image = Image.open('./dataset/imagenet/imagenet_1000_raw/n02782093_1.JPEG')
  dataset_load_time = time.time() - dataset_load_time


  size = classify.input_size(interpreter)
  image = image.convert('RGB').resize(size, Image.ANTIALIAS)
  classify.set_input(interpreter, image)
  start = time.perf_counter()
  interpreter.invoke()
  iter_times = time.perf_counter() - start
  classes = classify.get_output(interpreter, args.top_k, args.threshold)
  for klass in classes:
          accuracy = klass.score

   
  print('***** TF-lite matric *****')
  print('accuracy =', accuracy)
  print('model_load_time =', model_load_time)
  print('dataset_load_time =', dataset_load_time)
  print('inference_time =', iter_times)
  print('IPS =', 1 / (model_load_time + dataset_load_time + iter_times))
  print('IPS(inf) =', 1 / iter_times)


if __name__ == '__main__':
  main()
