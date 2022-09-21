#!/bin/bash


#image classification TF/FP32 model (mobilenet v1, mobilenet v2, inception v3)
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v1/mobilenet_v1.zip
unzip -q mobilenet_v1.zip && rm mobilenet_v1.zip

curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v2/mobilenet_v2.zip
unzip -q mobilenet_v2.zip && rm mobilenet_v2.zip

curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/inception_v3/inception_v3.zip
unzip -q inception_v3.zip && rm inception_v3.zip

#image classification EdgeTPU tflite INT8 model (mobilenet v1, mobilenet v2, inception v3)
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/classify.py
mkdir mobilenet_v1_edgetpu_tflite
## coral model
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v1_edgetpu_tflite/tf2_mobilenet_v1_1.0_224_ptq_edgetpu.tflite
mv tf2_mobilenet_v1_1.0_224_ptq_edgetpu.tflite ./mobilenet_v1_edgetpu_tflite

mkdir mobilenet_v2_edgetpu_tflite
## coral model
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v2_edgetpu_tflite/tf2_mobilenet_v2_1.0_224_ptq_edgetpu.tflite
mv tf2_mobilenet_v2_1.0_224_ptq_edgetpu.tflite ./mobilenet_v2_edgetpu_tflite
## custom model
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v2_edgetpu_tflite/custom-mobilenet_v2_edgetpu.tflite
mv custom-mobilenet_v2_edgetpu.tflite ./mobilenet_v2_edgetpu_tflite

mkdir inception_v3_edgetpu_tflite
## coral model
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/inception_v3_edgetpu_tflite/inceptionv3_edgetpu.tflite
mv inceptionv3_edgetpu.tflite ./inception_v3_edgetpu_tflite
## custom model
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/inception_v3_edgetpu_tflite/custom-inceptionv3_edgetpu.tflite
mv custom-inceptionv3_edgetpu.tflite ./inception_v3_edgetpu_tflite

#image classification tflite INT8 model (mobilenet v1, mobilenet v2, inception v3)
mkdir mobilenet_v1_quantization_tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v1_quantization_tflite/mobilenetv1.tflite
mv mobilenetv1.tflite ./mobilenet_v1_quantization_tflite

mkdir mobilenet_v2_quantization_tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v2_quantization_tflite/mobilenetv2.tflite
mv mobilenetv2.tflite ./mobilenet_v2_quantization_tflite

mkdir inception_v3_quantization_tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/inception_v3_quantization_tflite/inceptionv3.tflite
mv inceptionv3.tflite ./inception_v3_quantization_tflite

#object detection TF/FP32 model (yolo v5)
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/yolo_v5/yolo_v5.zip
unzip -q yolo_v5.zip && rm yolo_v5.zip

#object detection EdgeTPU tflite INT8 model (yolo v5)
mkdir yolo_v5_edgetpu_tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/yolo_v5_edgetpu_tflite/yolov5s-int8_edgetpu.tflite
mv yolov5s-int8_edgetpu.tflite ./yolo_v5_edgetpu_tflite 

#object detection tflite FP16/INT8 model (yolo v5)
mkdir yolo_v5_quantization_tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/yolo_v5_quantization_tflite/yolov5s-fp16.tflite
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/yolo_v5_quantization_tflite/yolov5s-int8.tflite
mv yolov5s-fp16.tflite yolov5s-int8.tflite ./yolo_v5_quantization_tflite
