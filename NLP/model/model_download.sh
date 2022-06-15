#!/bin/bash

curl -O https://edge-inference.s3.us-west-2.amazonaws.com/bert_imdb_model.h5
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/rnn_imdb_model.h5
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/lstm_imdb_model.h5
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_model.h5
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_batch_1.tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_batch_2.tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_batch_4.tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_batch_8.tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_batch_16.tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_batch_32.tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_batch_64.tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/distilbert_sst2_batch_128.tflite

