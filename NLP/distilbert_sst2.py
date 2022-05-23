# Install the required package
# !pip install datasets
# !pip install transformers

# Import Library
import datasets
from transformers import pipeline
from tqdm.auto import tqdm
import time
import pandas as pd
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

model = None
load_model_time = None
result_df = pd.DataFrame(columns=['batch_size', 'accuracy', 'load_model_time', 'load_dataset_time','total_inference_time', 'avg_inference_time','ips', 'ips_inf'])
X_test = None
y_test = None

def load_model():

  global load_model_time
  global model

  load_model_time = time.time()
  model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="tf", device=0)     # devices -1 : CPU, 0 : GPU
  load_model_time = time.time() - load_model_time

  return model

def load_test_batch(batch_size):

  global X_test
  global y_test

  dataset = datasets.load_dataset("glue", "sst2", split='validation')

  X_test = dataset[:len(dataset)]["sentence"]
  y_test = dataset[:len(dataset)]["label"]

  X_test_preprocess = []
  for i in range(len(X_test)):
    # 영어문장을 utf-8 로 인코딩한 뒤, ascii 로 디코딩
    # 이때, 오류가 발생하는 문자열은 무시하고 정상적인 문자열만 리턴 
    X_test_preprocess.append(X_test[i].encode('utf-8').decode('ascii', 'ignore'))

  test_batch = tf.data.Dataset.from_tensor_slices((X_test_preprocess, y_test)).batch(batch_size)

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
      
      X_test_batch = X_test_batch.numpy().astype('str').tolist()

      # 배치별 데이터 추론 시간
      # inference_time_per_batch = time.time()

      # 배치 단위별 데이터셋 분류
      y_pred_batch = model(X_test_batch)
      
      # 배치별 데이터 추론 시간 저장
      # iter_times.append(time.time() - inference_time_per_batch)

      # 배치 사이즈 만큼의 예측 라벨 저장
      pred_labels.extend([*y_pred_batch])
      
      # 배치 사이즈 만큼의 실제 라벨 저장
      real_labels.extend([*(y_test_batch.numpy().tolist())])      

      # 디버깅
      success += batch_size
      if (success % 500 == 0):
        print("{}/{}".format(success,len(test_batch)*batch_size))
  
  inference_time = time.time() - inference_time

  # 모든 데이터에 대한 배치별 추론 시간을 배열화
  # iter_times = np.array(iter_times)

  # 모든 데이터에 대한 실제라벨과 예측라벨을 비교한 뒤, 정확도 계산
  labeling = {'POSITIVE': 1, 'NEGATIVE': 0}
  accuracy = len([1 for pred, real in zip(pred_labels, real_labels) if labeling[pred['label']] == real ]) / len(real_labels)
  
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

# 모델명
model_name = 'distilbert_sst2'

# 모델 로드
load_model()

# 배치 단위로 추론 
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
  inference(batch_size)

# 배치 단위 추론 결과 데이터를 저장할 경로
result_csv=f'./csv/{model_name}_result.csv'

# 배치 단위 추론 결과 데이터 저장
result_df.to_csv(result_csv, index=False)
