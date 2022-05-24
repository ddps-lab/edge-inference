# 참고 : https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb?hl=it_IT#scrollTo=6o2a5ZIvRcJq
# 참고 : https://github.com/harenlin/IMDB-Sentiment-Analysis-Using-BERT-Fine-Tuning/blob/main/BERT_Fine_Tune.ipynb

# Install the required package
# !pip install bert-for-tf2
# !pip install tensorflow_hub

# Import modules
import os
import re
import pickle
import time
import bert
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tqdm import tqdm
from tensorflow.keras.models import load_model

# Check GPU Availability
device_name = tf.test.gpu_device_name()
if not device_name:
    print('Cannot found GPU. Training with CPU')
else:
    print('Found GPU at :{}'.format(device_name))


# 전역 변수 설정
model = None
load_model_time = None
x_train = None
x_valid = None
x_test = None
y_train = None
y_valid = None
y_test = None
MAX_SEQ_LEN = 500

result_df = pd.DataFrame(columns=['batch_size', 'accuracy', 'load_model_time', 'load_dataset_time','total_inference_time', 'avg_inference_time','ips', 'ips_inf'])

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = "positive"
  neg_df["polarity"] = "negative"
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)
  
  train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))
  
  return train_df, test_df

def create_tonkenizer(bert_layer):
    """Instantiate Tokenizer with vocab"""
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy() 
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
    print("Vocab size:", len(tokenizer.vocab))
    return tokenizer

def get_ids(tokens, tokenizer, MAX_SEQ_LEN):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
    return input_ids

def get_masks(tokens, MAX_SEQ_LEN):
    """Masks: 1 for real tokens and 0 for paddings"""
    return [1] * len(tokens) + [0] * (MAX_SEQ_LEN - len(tokens))

def get_segments(tokens, MAX_SEQ_LEN):
    """Segments: 0 for the first sequence, 1 for the second"""  
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (MAX_SEQ_LEN - len(tokens))

def create_single_input(sentence, tokenizer, max_len):
    """Create an input from a sentence"""
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:max_len] # max_len = MAX_SEQ_LEN - 2, why -2 ? ans: reserved for [CLS] & [SEP]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    return get_ids(stokens, tokenizer, max_len+2), get_masks(stokens, max_len+2), get_segments(stokens, max_len+2)
  
def convert_sentences_to_features(sentences, tokenizer, MAX_SEQ_LEN):
    """Convert sentences to features: input_ids, input_masks and input_segments"""
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences, position=0, leave=True):
      ids, masks, segments = create_single_input(sentence, tokenizer, MAX_SEQ_LEN-2) # why -2 ? ans: reserved for [CLS] & [SEP]
      input_ids.append(ids)
      input_masks.append(masks)
      input_segments.append(segments)
    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]

def nlp_model(bert_base):
    
    # Load the pre-trained BERT base model
    bert_layer = hub.KerasLayer(handle=bert_base, trainable=True)  
    # BERT layer three inputs: ids, masks and segments
    input_ids = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")           
    input_masks = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_masks")       
    input_segments = Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="segment_ids")

    inputs = [input_ids, input_masks, input_segments] # BERT inputs
    pooled_output, sequence_output = bert_layer(inputs) # BERT outputs
    
    x = Dense(units=768, activation='relu')(pooled_output) # hidden layer 
    x = Dropout(0.15)(x) 
    outputs = Dense(2, activation="softmax")(x) # output layer

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 저장되어있는 데이터가 없다면, 데이터를 전처리한 뒤 저장
def preprocess_and_save_data(saved_dataset_dir):

  global x_train
  global x_valid
  global x_test
  global y_train
  global y_valid
  global y_test

  global model
  
  train_df, test_df = download_and_load_datasets()
  df = pd.concat([train_df, test_df])
  df = df.drop("sentiment", axis=1)
  df.rename(columns = {'sentence': 'review', 'polarity' : 'sentiment'}, inplace=True)

  # hyper-parameters
  MAX_SEQ_LEN = 500

  # model construction (we construct model first inorder to use bert_layer's tokenizer)
  bert_base = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
  model = nlp_model(bert_base) 

  # Create examples for training and testing
  df = df.sample(frac=1) # Shuffle the dataset
  # we would like to use bert tokenizer; 
  # therefore, chech model.summary() and find the index of bert_layer
  tokenizer = create_tonkenizer(model.layers[3])

  # create training data and testing data
  x_train = convert_sentences_to_features(df['review'][:20000], tokenizer, MAX_SEQ_LEN)
  x_valid = convert_sentences_to_features(df['review'][20000:25000], tokenizer, MAX_SEQ_LEN)
  x_test = convert_sentences_to_features(df['review'][25000:], tokenizer, MAX_SEQ_LEN)
  df['sentiment'].replace('positive', 1., inplace=True)
  df['sentiment'].replace('negative', 0., inplace=True)
  one_hot_encoded = to_categorical(df['sentiment'].values)
  y_train = one_hot_encoded[:20000]
  y_valid = one_hot_encoded[20000:25000]
  y_test =  one_hot_encoded[25000:]

  # 데이터 전처리 결과 저장
  with open(saved_dataset_dir+'_x_train.pkl','wb') as f:
    pickle.dump(x_train, f)
  with open(saved_dataset_dir+'_x_valid.pkl','wb') as f:
    pickle.dump(x_valid, f)
  with open(saved_dataset_dir+'_x_test.pkl','wb') as f:
    pickle.dump(x_test, f)
  with open(saved_dataset_dir+'_y_train.pkl','wb') as f:
    pickle.dump(y_train, f)
  with open(saved_dataset_dir+'_y_valid.pkl','wb') as f:
    pickle.dump(y_valid, f)
  with open(saved_dataset_dir+'_y_test.pkl','wb') as f:
    pickle.dump(y_test, f)


# 저장되어있는 모델이 없다면, 모델을 새롭게 훈련한 뒤 저장
def train_and_save_model(saved_model_dir):
  
  global model
  global x_train
  global x_valid
  global y_train
  global y_valid
  
  with open(saved_dataset_dir+'_x_train.pkl','rb') as f:
    x_train = pickle.load(f)
  with open(saved_dataset_dir+'_x_valid.pkl','rb') as f:
    x_valid = pickle.load(f)
  with open(saved_dataset_dir+'_y_train.pkl','rb') as f:
    y_train = pickle.load(f)
  with open(saved_dataset_dir+'_y_valid.pkl','rb') as f:
    y_valid = pickle.load(f)

    bert_base = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    model = nlp_model(bert_base)

  BATCH_SIZE = 8
  EPOCHS = 1

  # use adam optimizer to minimize the categorical_crossentropy loss
  optimizer = Adam(learning_rate=2e-5)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  # fit the data to the model
  history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)   

  model.save(saved_model_dir, include_optimizer=False, save_format='tf')
  
# 모델 로드
def load_model(saved_model_dir):

  global load_model_time
  global model
  
  load_model_time = time.time()
  model = tf.keras.models.load_model(saved_model_dir, custom_objects={'KerasLayer': hub.KerasLayer})
  load_model_time = time.time() - load_model_time

# 테스트 데이터를 배치 단위로 제공
def load_test_batch(batch_size):

  global x_test
  global y_test

  with open(saved_dataset_dir+'_x_test.pkl','rb') as f:
    x_test = pickle.load(f)
  with open(saved_dataset_dir+'_y_test.pkl','rb') as f:
    y_test = pickle.load(f)
  
  test_batch = tf.data.Dataset.from_tensor_slices(((x_test[0],x_test[1],x_test[2]),y_test)).batch(batch_size)

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
      
      # # 배치 사이즈 만큼의 실제 라벨 저장
      real_labels.extend(np.argmax(y_pred_batch.numpy(), axis=1))
      # # 배치 사이즈 만큼의 예측 라벨 저장
      pred_labels.extend(np.argmax(y_test_batch.numpy(), axis=1))

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
                                'avg_inference_time' : round(inference_time / len(x_test[0]), 4),
                                'ips' : round(len(x_test[0]) / (load_model_time + load_dataset_time + inference_time), 4), 
                                'ips_inf' : round(len(x_test[0]) / inference_time, 4)}, ignore_index=True)
    
  # 배치 단위 추론 결과 데이터 저장
  result_df.to_csv(result_csv, index=False)

model_name = 'bert_imdb'

# 데이터셋 저장할 경로
saved_dataset_dir=f'./dataset/{model_name}_dataset'

# 모델이 저장되어있는/저장할 경로
saved_model_dir=f'./model/{model_name}_model.h5'

# 배치 단위 추론 결과 데이터를 저장할 경로
result_csv=f'./csv/{model_name}_result.csv'

# 저장되어있는 데이터가 없다면, 데이터를 전처리한 뒤 저장
# preprocess_and_save_data(saved_dataset_dir)

# 저장되어있는 모델이 없다면, 모델을 새롭게 훈련한 뒤 저장
# train_and_save_model(saved_model_dir)

# 저장되어있는 모델이 있다면, 로드
load_model(saved_model_dir)

# 배치 단위로 추론 
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
  inference(batch_size)
