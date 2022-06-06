import tensorflow as tf
import time
import numpy as np
import pandas as pd

def tflite_converter(batch_size):

  # model_tensor_fix
  hdf5_path = f'./model/{model_name}_model.h5'
  model = tf.keras.models.load_model(hdf5_path)

  run_model = tf.function(lambda x: model(x))

  concrete_func = run_model.get_concrete_function(
      tf.TensorSpec([batch_size, 80], model.inputs[0].dtype))

  saved_model_path = f'./model/{model_name}_model_batch_{batch_size}'
  model.save(saved_model_path, save_format="tf", signatures=concrete_func)

  # tflite_converter
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  tflite_model = converter.convert()

  tflite_path = f'./model/{model_name}_batch_{batch_size}.tflite'
  open(tflite_path, "wb").write(tflite_model)

def load_tflite_model(batch_size):

  global load_model_time
  global model
  
  load_model_time = time.time()
  tflite_path = f'./model/{model_name}_batch_{batch_size}.tflite'

  model = tf.lite.Interpreter(model_path=tflite_path)
  model.allocate_tensors()

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

  input_details = model.get_input_details()
  output_details = model.get_output_details()

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
    X_test_len = len(X_test)-(len(X_test) % batch_size)
  
  # 디버깅용 변수
  success = 0

  # 전체 데이터에 대한 추론 시작
  inference_time = time.time()
  # 전체 데이터를 배치 단위로 묶어서 사용 (반복문 한번당 배치 단위 추론 한번)
  for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
      
      # 전체 데이터를 배치 단위로 나눈 뒤, 남은 나머지 데이터는 처리하지 않음
      if (X_test_batch.shape[0] == batch_size):
        
        X_test_batch = tf.cast(X_test_batch, tf.float32)

        model.set_tensor(input_details[0]['index'], X_test_batch)
        model.invoke()

        # 배치 단위별 데이터셋 분류
        y_pred_batch = model.get_tensor(output_details[0]['index'])

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
result_df = pd.DataFrame(columns=['batch_size', 'accuracy', 'load_model_time', 'load_dataset_time','total_inference_time', 'avg_inference_time','ips', 'ips_inf'])

model_name = 'lstm_imdb'
result_csv=f'./csv/{model_name}_result.csv'

# 배치 단위로 추론
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
  tflite_converter(batch_size)
  load_tflite_model(batch_size)
  inference(batch_size)
