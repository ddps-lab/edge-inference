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

def tflite_converter(batch_size):

  # Load HDF5
  hdf5_path = f'./model/{model_name}_model.h5'
  model = tf.keras.models.load_model(hdf5_path,custom_objects={'TFDistilBertModel': transformers.TFDistilBertModel})

  # Fix model tensor
  run_model = tf.function(lambda x: model(x))
  input_spec = [tf.TensorSpec([batch_size, 128], tf.int32), tf.TensorSpec([batch_size, 128], tf.int32)]
  concrete_func = run_model.get_concrete_function(input_spec)

  # Save SavedModel
  saved_model_path = f'./model/{model_name}_model_batch_{batch_size}'
  model.save(saved_model_path, save_format="tf", signatures=concrete_func)

  # Tflite Converter
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
  
  # default
  if (quantization=='fp32'):  
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
  
  # FP16 Configure
  elif (quantization=='fp16'):
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

  # Save tflite
  tflite_path = f'./model/{model_name}_{quantization}_model_batch_{batch_size}.tflite'
  open(tflite_path, "wb").write(tflite_model)
  
def load_tflite_model(batch_size):

  global load_model_time
  global model
  
  load_model_time = time.time()
  tflite_path = f'./model/{model_name}_{quantization}_model_batch_{batch_size}.tflite'

  model = tf.lite.Interpreter(model_path=tflite_path)
  model.allocate_tensors()

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

  bert_input_index = model.get_input_details()[0]["index"]
  bert_input_masks_index = model.get_input_details()[1]["index"]
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
    X_test_len = len(X_test)-(len(X_test)%batch_size)

  # 디버깅용 변수
  success = 0

  # 전체 데이터에 대한 추론 시작
  inference_time = time.time()
  # 전체 데이터를 배치 단위로 묶어서 사용 (반복문 한번당 배치 단위 추론 한번)
  for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
      
      # 전체 데이터를 배치 단위로 나눈 뒤, 남은 나머지 데이터는 처리하지 않음
      if (len(X_test_batch[0].numpy()) == batch_size):

        X_test_batch = tf.cast(X_test_batch, tf.int32)

        model.set_tensor(bert_input_index, X_test_batch[0])
        model.set_tensor(bert_input_masks_index, X_test_batch[1])
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

quantization = 'fp16'
model_name = 'distilbert_sst2'
saved_model_dir=f'./model/{model_name}_model.h5'
result_csv=f'./csv/{model_name}_{quantization}_model_result.csv'

# 배치 단위별 모델 생성
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
  tflite_converter(batch_size)

# 배치 단위로 추론
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
  load_tflite_model(batch_size)
  inference(batch_size)
