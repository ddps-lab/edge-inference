import tensorflow as tf
import numpy as np
import time
import pandas as pd


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=15000)
  
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                       value= 0,
                                                       padding = 'pre',
                                                       maxlen = 130)
print('X_test', X_test)

batch_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)

load_dataset_time = time.time()
for i, (X_batch_data, y_batch_data) in enumerate(batch_data):
    raw_data=X_batch_data
    break
load_dataset_time = time.time() - load_dataset_time
print(raw_data)


model_name = 'rnn_imdb'
saved_model_dir=f'./model/{model_name}_model.h5'

load_model_start_time = time.time()
model = tf.keras.models.load_model(saved_model_dir)
load_model_time = time.time() - load_model_start_time


inference_time = time.time()
pred_labels = []
real_labels = []

raw_inference_start = time.time()
y_pred = model(raw_data)
raw_inference_time = time.time() - raw_inference_start

real_labels.extend(y_batch_data.numpy())
y_pred = np.where(y_pred > 0.5, 1, 0)
y_pred = y_pred.reshape(-1)
pred_labels.extend(y_pred)

accuracy = np.sum(np.array(real_labels) == np.array(pred_labels))/len(real_labels)
inference_time = time.time() - inference_time

print('pred_labels', pred_labels)
print('real_labels', real_labels)

print('accuracy' , accuracy)
print('load_model_time', load_model_time)
print('load_dataset_time' , load_dataset_time)
print('total_inference_time', inference_time)
print('raw_inference_time', raw_inference_time / len(pred_labels))
print('ips' , len(pred_labels) / (load_model_time + load_dataset_time + inference_time))
print('ips(inf)' , len(pred_labels) / inference_time)
