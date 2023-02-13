#!/bin/bash

curl -O https://edge-inference.s3.us-west-2.amazonaws.com/NLP/bert_dataset.zip
unzip bert_dataset.zip && rm -rf bert_dataset.zip
