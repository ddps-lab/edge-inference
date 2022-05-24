# Model: LSTM
# Dataset: imdb

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
result_df = pd.DataFrame(columns=['batch_size', 'accuracy', 'load_model_time', 'load_dataset_time','total_inference_time', 'avg_inference_time','ips', 'ips_inf'])


# 모델 훈련 및 저장
def train_and_save_model(saved_model_dir):
  
  global model
  
  num_words = 20000
  maxlen = 80

  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
  
  X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                        value = 0,
                                                        padding = 'pre',
                                                        maxlen = maxlen)
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Embedding(num_words, 128))
  model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy',
                optimizer="rmsprop",
                metrics=['accuracy'])

  history = model.fit(X_train,
                      y_train,
                      validation_split=0.2,
                      epochs = 5,
                      batch_size = 128,
                      verbose = 1)   

  model.save(saved_model_dir, include_optimizer=False, save_format='tf')


  
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

  num_words = 20000
  maxlen = 80

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

  # 배치 단위에 따른 추론 시간 저장
  # iter_times = []
  
  # 배치 단위의 테스트 데이터 로드
  load_dataset_time = time.time()
  test_batch = load_test_batch(batch_size)
  load_dataset_time = time.time() - load_dataset_time

  # 디버깅용 변수
  success = 0

  # 전체 데이터에 대한 추론 시작
  inference_time = time.time()
  # 전체 데이터를 배치 단위로 묶어서 사용 (반복문 한번당 배치 단위 추론 한번)
  for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
      
      # 배치별 데이터 추론 시간
      # inference_time_per_batch = time.time()

      # 배치 단위별 데이터셋 분류
      y_pred_batch = model(X_test_batch)
      
      # 배치별 데이터 추론 시간 저장
      # iter_times.append(time.time() - inference_time_per_batch)
      
      # 배치 사이즈 만큼의 실제 라벨 저장
      real_labels.extend(y_test_batch.numpy())
      # 배치 사이즈 만큼의 예측 라벨 저장
      y_pred_batch = np.where(y_pred_batch > 0.5, 1, 0)
      y_pred_batch = y_pred_batch.reshape(-1)
      pred_labels.extend(y_pred_batch)

      # 디버깅
      success += batch_size
      if (success % 500 == 0):
        print("{}/{}".format(success,len(test_batch)*batch_size))
  
  inference_time = time.time() - inference_time

  # 모든 데이터에 대한 배치별 추론 시간을 배열화
  # iter_times = np.array(iter_times)

  # 모든 데이터에 대한 실제라벨과 예측라벨을 비교한 뒤, 정확도 계산
  accuracy = np.sum(np.array(real_labels) == np.array(pred_labels))/len(real_labels)
  
  # Metric 결과 저장
  global result_df
  result_df = result_df.append({'batch_size' : batch_size , 
                                'accuracy' : accuracy, 
                                'load_model_time' : round(load_model_time, 4), 
                                'load_dataset_time' : round(load_dataset_time, 4),
                                'total_inference_time' : round(inference_time, 4), 
                                'avg_inference_time' : round(inference_time / len(X_test), 4),
                                'ips' : round(len(X_test) / (load_model_time + load_dataset_time + inference_time), 4), 
                                'ips_inf' : round(len(X_test) / inference_time, 4)}, ignore_index=True)
    
  # 배치 단위 추론 결과 데이터 저장
  result_df.to_csv(result_csv, index=False)


# 모델이 저장되어있는/저장할 경로
model_name = 'lstm_imdb'
saved_model_dir=f'./model/{model_name}_model'

# 배치 단위 추론 결과 데이터를 저장할 경로
result_csv=f'./csv/{model_name}_result.csv'

# 저장되어있는 모델이 없다면, 모델을 새롭게 학습한 뒤 저장
# train_and_save_model(saved_model_dir)

# 저장되어있는 모델이 있다면, 로드
load_model(saved_model_dir)

# 배치 단위로 추론 
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
  inference(batch_size)
