import transformers
import datasets
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import tqdm


def create_bert_input_features(tokenizer, docs, max_seq_length):

    all_ids, all_masks = [], []
    for doc in tqdm.tqdm(docs, desc="Converting docs to features"):
        tokens = tokenizer.tokenize(doc)
        if len(tokens) > max_seq_length-2:
            tokens = tokens[0 : (max_seq_length-2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(ids)
        while len(ids) < max_seq_length:
            ids.append(0)
            masks.append(0)
        all_ids.append(ids)
        all_masks.append(masks)
    encoded = np.array([all_ids, all_masks])
    return encoded



def load_test_batch(batch_size):

  X_test = None
  y_test=None

  dataset = datasets.load_dataset("glue", "sst2")

  X_test = np.array(dataset['validation']["sentence"])
  y_test = np.array(dataset['validation']["label"])


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


model_name = 'distilbert_sst2'
saved_model_dir=f'./model/{model_name}_model.h5'
load_model_time = time.time()
model = tf.keras.models.load_model(saved_model_dir,custom_objects={'TFDistilBertModel': transformers.TFDistilBertModel})
load_model_time = time.time() - load_model_time


batch_size=1
valid_ds = load_test_batch(batch_size)

load_dataset_time = time.time()
for i, (X_test_batch, y_test_batch) in enumerate(valid_ds):
    raw_data=X_test_batch
    break
load_dataset_time = time.time() - load_dataset_time


pred_labels = []
real_labels = []

load_dataset_time = time.time()
test_batch = load_test_batch(batch_size)
load_dataset_time = time.time() - load_dataset_time

inference_time = time.time()
raw_inference_time = time.time()
y_pred_batch = model(X_test_batch)
raw_inference_time = time.time() - raw_inference_time
real_labels.extend(y_test_batch.numpy())
y_pred_batch = np.where(y_pred_batch > 0.5, 1, 0)
y_pred_batch = y_pred_batch.reshape(-1)
pred_labels.extend(y_pred_batch)
accuracy = np.sum(np.array(real_labels) == np.array(pred_labels))/len(real_labels)
inference_time = time.time() - inference_time

print('accuracy', accuracy)
print('load_model_time', load_model_time)
print('load_dataset_time' , load_dataset_time)
print('total_inference_time', inference_time)
print('raw_inference_time', raw_inference_time / len(pred_labels))
print('ips' , len(pred_labels) / (load_model_time + load_dataset_time + inference_time))
print('ips(inf)' , len(pred_labels) / inference_time)
