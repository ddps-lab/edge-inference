import os
import re
import pickle
import time
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

MAX_SEQ_LEN = 500

def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)


def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = "positive"
  neg_df["polarity"] = "negative"
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


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
    stokens = stokens[:max_len] 
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    return get_ids(stokens, tokenizer, max_len+2), get_masks(stokens, max_len+2), get_segments(stokens, max_len+2)

def convert_sentences_to_features(sentences, tokenizer, MAX_SEQ_LEN):
    """Convert sentences to features: input_ids, input_masks and input_segments"""
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences, position=0, leave=True):
      ids, masks, segments = create_single_input(sentence, tokenizer, MAX_SEQ_LEN-2) 
      input_ids.append(ids)
      input_masks.append(masks)
      input_segments.append(segments)
    return [np.asarray(input_ids, dtype=np.int32), np.asarray(input_masks, dtype=np.int32), np.asarray(input_segments, dtype=np.int32)]



def load_test_batch(batch_size):

  x_test = None
  y_test = None

  with open(saved_dataset_dir+'_x_test.pkl','rb') as f:
    x_test = pickle.load(f)
  with open(saved_dataset_dir+'_y_test.pkl','rb') as f:
    y_test = pickle.load(f)

  test_batch = tf.data.Dataset.from_tensor_slices(((x_test[0],x_test[1],x_test[2]),y_test)).batch(batch_size)

  return test_batch


pred_labels = []
real_labels = []

model_name = 'bert_imdb'
saved_model_dir=f'./model/{model_name}_model.h5'
load_model_time = time.time()
model = tf.keras.models.load_model(saved_model_dir, custom_objects={'KerasLayer': hub.KerasLayer})
load_model_time = time.time() - load_model_time

batch_size=1
saved_dataset_dir=f'./dataset/{model_name}_dataset'
test_batch = load_test_batch(batch_size)

load_dataset_time = time.time()
for i, (X_test_batch, y_test_batch) in enumerate(test_batch):
    raw_data=X_test_batch
    break
load_dataset_time = time.time() - load_dataset_time


inference_time = time.time()
raw_inference_time = time.time()
y_pred_batch = model(raw_data)
raw_inference_time = time.time() - raw_inference_time
real_labels.extend(np.argmax(y_pred_batch.numpy(), axis=1))
pred_labels.extend(np.argmax(y_test_batch.numpy(), axis=1))
accuracy = np.sum(np.array(real_labels) == np.array(pred_labels))/len(real_labels)
inference_time = time.time() - inference_time


print('accuracy',accuracy)
print('load_model_time', load_model_time)
print('load_dataset_time' , load_dataset_time)
print('total_inference_time', inference_time)
print('raw_inference_time', raw_inference_time / len(pred_labels))
print('ips' , len(pred_labels) / (load_model_time + load_dataset_time + inference_time))
print('ips(inf)' , len(pred_labels) / inference_time)
