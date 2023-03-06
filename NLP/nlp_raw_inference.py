import tensorflow as tf
import numpy as np
import time
import pandas as pd

X_test = None


  
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
  
batch_size=1
test_batch = load_test_batch(batch_size)

load_dataset_time = time.time()
for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
    raw_data=X_test_batch
    break
load_dataset_time = time.time() - load_dataset_time

print(raw_data)

model = None
load_model_time = None


# 모델 로드
def load_model(saved_model_dir):

  global load_model_time
  global model

  load_model_time = time.time()
  model = tf.keras.models.load_model(saved_model_dir)
  load_model_time = time.time() - load_model_time


# 모델이 저장되어있는/저장할 경로
model_name = 'rnn_imdb'
saved_model_dir=f'./model/{model_name}_model.h5'

# 저장되어있는 모델이 있다면, 로드
load_model_time = time.time()
load_model(saved_model_dir)
load_model_time = time.time() - load_model_time

inference_time = time.time()
pred_labels = []
real_labels = []

raw_inference_start = time.time()
y_pred_batch = model(raw_data)
raw_inference_time = time.time() - raw_inference_start

# 배치 사이즈 만큼의 실제 라벨 저장
real_labels.extend(y_test_batch.numpy())
# 배치 사이즈 만큼의 예측 라벨 저장
y_pred_batch = np.where(y_pred_batch > 0.5, 1, 0)
y_pred_batch = y_pred_batch.reshape(-1)
pred_labels.extend(y_pred_batch)

accuracy = np.sum(np.array(real_labels) == np.array(pred_labels))/len(real_labels)
inference_time = time.time() - inference_time

print('accuracy' , accuracy)
print('load_model_time', load_model_time)
print('load_dataset_time' , load_dataset_time)
print('total_inference_time', inference_time)
print('raw_inference_time', raw_inference_time / len(pred_labels))
print('ips' , len(pred_labels) / (load_model_time + load_dataset_time + inference_time))
print('ips(inf)' , len(pred_labels) / inference_time)
