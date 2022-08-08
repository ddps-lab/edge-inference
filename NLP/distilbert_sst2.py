# Install the required package
# !pip install datasets
# !pip install transformers

import transformers
import datasets
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import tqdm

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  
def create_bert_input_features(tokenizer, docs, max_seq_length):
    
    all_ids, all_masks = [], []
    for doc in tqdm.tqdm(docs, desc="Converting docs to features"):
        tokens = tokenizer.tokenize(doc)
        if len(tokens) > max_seq_length-2:
            tokens = tokens[0 : (max_seq_length-2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(ids)
        # Zero-pad up to the sequence length.
        while len(ids) < max_seq_length:
            ids.append(0)
            masks.append(0)
        all_ids.append(ids)
        all_masks.append(masks)
    encoded = np.array([all_ids, all_masks])
    return encoded

def train_and_save_model(saved_model_dir):

  global model

  dataset = datasets.load_dataset("glue", "sst2")

  train_reviews = np.array(dataset['train']["sentence"])
  train_sentiments = np.array(dataset['train']["label"])

  valid_reviews = np.array(dataset['validation']["sentence"])
  valid_sentiments = np.array(dataset['validation']["label"])

  # test_reviews = np.array(dataset['test']["sentence"])

  tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

  MAX_SEQ_LENGTH = 128

  train_features_ids, train_features_masks = create_bert_input_features(tokenizer, train_reviews, 
                                                                        max_seq_length=MAX_SEQ_LENGTH)
  
  val_features_ids, val_features_masks = create_bert_input_features(tokenizer, valid_reviews, 
                                                                    max_seq_length=MAX_SEQ_LENGTH)

  train_ds = (
    tf.data.Dataset
    .from_tensor_slices(((train_features_ids, train_features_masks), train_sentiments))
    .shuffle(2048)
    .batch(128)
    .prefetch(tf.data.experimental.AUTOTUNE)
  )

  valid_ds = (
    tf.data.Dataset
    .from_tensor_slices(((val_features_ids, val_features_masks), valid_sentiments))
    .batch(128)
    .prefetch(tf.data.experimental.AUTOTUNE)
  )

  inp_id = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name="bert_input_ids")
  inp_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH,), dtype='int32', name="bert_input_masks")
  inputs = [inp_id, inp_mask]

  hidden_state = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')(inputs)[0]
  pooled_output = hidden_state[:, 0]    
  dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled_output)
  drop1 = tf.keras.layers.Dropout(0.25)(dense1)
  dense2 = tf.keras.layers.Dense(256, activation='relu')(drop1)
  drop2 = tf.keras.layers.Dropout(0.25)(dense2)
  output = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, 
                                              epsilon=1e-08), 
                loss='binary_crossentropy', metrics=['accuracy'])
  
  model.fit(train_ds, 
      validation_data=valid_ds,
      epochs=3)

  model.save(saved_model_dir, include_optimizer=False, save_format='tf')

# 모델 로드
def load_model(batch_size):

  global load_model_time
  global model
  
  load_model_time = time.time()
  model = tf.keras.models.load_model(saved_model_dir,custom_objects={'TFDistilBertModel': transformers.TFDistilBertModel})
  load_model_time = time.time() - load_model_time

# 테스트 데이터를 배치 단위로 제공
def load_test_batch(batch_size):
  
  global X_test
  global y_test
  
  dataset = datasets.load_dataset("glue", "sst2")

  X_test = np.array(dataset['validation']["sentence"])
  y_test = np.array(dataset['validation']["label"])

  # test_reviews = np.array(dataset['test']["sentence"])

  tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

  MAX_SEQ_LENGTH = 128

  val_features_ids, val_features_masks = create_bert_input_features(tokenizer, X_test, 
                                                                    max_seq_length=MAX_SEQ_LENGTH)
  valid_ds = (
    tf.data.Dataset
    .from_tensor_slices(((val_features_ids, val_features_masks), y_test))
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
  )

  return valid_ds

def inference(batch_size):	

  # 전체 데이터에 대한 예측라벨 및 실제라벨 저장
  pred_labels = []
  real_labels = []
  
  # 배치 단위의 테스트 데이터 로드
  load_dataset_time = time.time()
  test_batch = load_test_batch(batch_size)
  load_dataset_time = time.time() - load_dataset_time

  # 전체 데이터가 배치 단위에 맞게 나눠지는 경우
  if (len(X_test) % batch_size == 0):
    X_test_len = len(X_test)
  # 전체 데이터가 배치 단위에 맞게 나눠지지 않는 경우
  else:
    X_test_len = len(X_test)-(len(X_test)%batch_size)

  # 디버깅용 변수
  success = 0

  # 전체 데이터에 대한 추론 시작
  inference_time = time.time()
  # 전체 데이터를 배치 단위로 묶어서 사용 (반복문 한번당 배치 단위 추론 한번)
  for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
      
      # 전체 데이터를 배치 단위로 나눈 뒤, 남은 나머지 데이터는 처리하지 않음
      if (len(X_test_batch[0].numpy()) == batch_size):

        # 배치 단위별 데이터셋 분류
        y_pred_batch = model(X_test_batch)
      
        # 배치 사이즈 만큼의 실제 라벨 저장
        real_labels.extend(y_test_batch.numpy())
        # 배치 사이즈 만큼의 예측 라벨 저장
        y_pred_batch = np.where(y_pred_batch > 0.5, 1, 0)
        y_pred_batch = y_pred_batch.reshape(-1)
        pred_labels.extend(y_pred_batch)

        # 디버깅
        success += batch_size
        if (success % 100 == 0):
          print("{}/{}".format(success,X_test_len))
  
  inference_time = time.time() - inference_time

  # 모든 데이터에 대한 실제라벨과 예측라벨을 비교한 뒤, 정확도 계산
  accuracy = np.sum(np.array(real_labels) == np.array(pred_labels))/len(real_labels)
  
  # Metric 결과 저장
  global result_df
  result_df = result_df.append({'batch_size' : batch_size , 
                                'accuracy' : accuracy, 
                                'load_model_time' : round(load_model_time, 4), 
                                'load_dataset_time' : round(load_dataset_time, 4),
                                'total_inference_time' : round(inference_time, 4), 
                                'avg_inference_time' : round(inference_time / X_test_len, 4),
                                'ips' : round(X_test_len / (load_model_time + load_dataset_time + inference_time), 4), 
                                'ips_inf' : round(X_test_len / inference_time, 4)}, ignore_index=True)
  # 배치 단위 추론 결과 데이터 저장
  result_df.to_csv(result_csv, index=False)

# 전역 변수 설정
model = None
load_model_time = None
X_test = None
y_test = None
result_df = pd.DataFrame(columns=['batch_size', 'accuracy', 'load_model_time', 'load_dataset_time','total_inference_time', 'avg_inference_time','ips', 'ips_inf'])

# 모델이 저장되어있는/저장할 경로
model_name = 'distilbert_sst2'
saved_model_dir=f'./model/{model_name}_model.h5'

# 배치 단위 추론 결과 데이터를 저장할 경로
result_csv=f'./csv/{model_name}_result.csv'

# 저장되어있는 모델이 없다면, 모델을 새롭게 학습한 뒤 저장
# train_and_save_model(saved_model_dir)

# 저장되어있는 모델이 있다면 로드
load_model(saved_model_dir)

# 배치 단위로 추론
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
  inference(batch_size)
