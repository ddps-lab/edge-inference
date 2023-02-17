import tensorflow as tf
import numpy as np
import time
import pandas as pd

# Check GPU Availability
device_name = tf.test.gpu_device_name()
if not device_name:
    print('Cannot found GPU. Training with CPU')
else:
    print('Found GPU at :{}'.format(device_name))

# 전역 변수 설정
model = None
load_model_time = None
X_test = None

  
# 모델 로드
def load_model(saved_model_dir):

  global load_model_time
  global model
  
  load_model_time = time.time()
  model = tf.keras.models.load_model(saved_model_dir)
  load_model_time = time.time() - load_model_time

# 테스트 데이터를 배치 단위로 제공
def load_test_batch(batch_size):
  
  global X_test
  
  num_words = 15000
  maxlen = 130

  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
  
  X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                        value= 0,
                                                        padding = 'pre',
                                                        maxlen = maxlen )
  
  test_batch = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

  return test_batch

def inference(batch_size):	

  # 전체 데이터에 대한 예측라벨 및 실제라벨 저장
  pred_labels = []
  real_labels = []

  
  # 배치 단위의 테스트 데이터 로드
  load_dataset_time = time.time()
  test_batch = load_test_batch(batch_size)
  load_dataset_time = time.time() - load_dataset_time

  # 전체 데이터에 대한 추론 시작
  inference_time = time.time()
  # 전체 데이터를 배치 단위로 묶어서 사용 (반복문 한번당 배치 단위 추론 한번)
  for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
      
      # 배치 단위별 데이터셋 분류
      raw_inference_start = time.time()
      y_pred_batch = model(X_test_batch)
      raw_inference_time = time.time() - raw_inference_start
    
      # 배치 사이즈 만큼의 실제 라벨 저장
      real_labels.extend(y_test_batch.numpy())
      # 배치 사이즈 만큼의 예측 라벨 저장
      y_pred_batch = np.where(y_pred_batch > 0.5, 1, 0)
      y_pred_batch = y_pred_batch.reshape(-1)
      pred_labels.extend(y_pred_batch)

      break 
  inference_time = time.time() - inference_time

  # 모든 데이터에 대한 실제라벨과 예측라벨을 비교한 뒤, 정확도 계산
  accuracy = np.sum(np.array(real_labels) == np.array(pred_labels))/len(real_labels)
  
  print('accuracy' , accuracy) 
  print('load_model_time', load_model_time) 
  print('load_dataset_time' , load_dataset_time)
  print('total_inference_time', inference_time) 
  print('raw_inference_time', raw_inference_time / len(pred_labels))
  print('ips' , len(pred_labels) / (load_model_time + load_dataset_time + inference_time))
  print('ips(inf)' , len(pred_labels) / inference_time)

# 모델이 저장되어있는/저장할 경로
model_name = 'rnn_imdb'
saved_model_dir=f'./model/{model_name}_model.h5'

# 저장되어있는 모델이 있다면, 로드
load_model(saved_model_dir)

# 배치 단위로 추론
inference(1)
