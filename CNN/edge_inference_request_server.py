import argparse

from model.yolov5.models.common import DetectMultiBackend
from model.yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from model.yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                                        colorstr, cv2,
                                        increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer,
                                        xyxy2xywh)
import torch
from pathlib import Path

from flask import Flask
import tensorflow as tf
import numpy as np
import shutil
import os
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(
            memory_limit=0.6 * 1024)])

from tensorflow.keras.applications import (
    mobilenet,
    mobilenet_v2,
    inception_v3
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mobilenet,mobilenet_v2,inception_v3', type=str)
parser.add_argument('--hostname', default='0.0.0.0', type=str)
parser.add_argument('--port', default=5001, type=int)
args = parser.parse_args()
models_to_load = args.model.split(',')
hostname = args.hostname
port = args.port

models = {
    'mobilenet': mobilenet,
    'mobilenet_v2': mobilenet_v2,
    'inception_v3': inception_v3
}

models_detail = {
    'mobilenet': mobilenet.MobileNet(weights='imagenet'),
    'mobilenet_v2': mobilenet_v2.MobileNetV2(weights='imagenet'),
    'inception_v3': inception_v3.InceptionV3(weights='imagenet')
}


def mobilenet_load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[224, 224])


def inception_load_image(image_path):
    return tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=[299, 299])


def image_to_array(image):
    return tf.keras.preprocessing.image.img_to_array(image, dtype=np.int32)


def image_preprocess(image_array, model_name):
    return models[model_name].preprocess_input(
        image_array[tf.newaxis, ...])


print('\npreprossing images...')

mobilenetv1_image_path = './dataset/imagenet/imagenet_1000_raw/n02782093_1.JPEG'
mobilenetv2_image_path = './dataset/imagenet/imagenet_1000_raw/n04404412_1.JPEG'
inceptionv3_image_path = './dataset/imagenet/imagenet_1000_raw/n13040303_1.JPEG'

mobilenetv1_test_image = mobilenet_load_image(mobilenetv1_image_path)
mobilenetv2_test_image = mobilenet_load_image(mobilenetv2_image_path)
inceptionv3_test_image = inception_load_image(inceptionv3_image_path)

mobilenetv1_test_image_array = image_to_array(mobilenetv1_test_image)
mobilenetv2_test_image_array = image_to_array(mobilenetv2_test_image)
inceptionv3_test_image_array = image_to_array(inceptionv3_test_image)

mobilenetv1_test_image_preprocessed = image_preprocess(mobilenetv1_test_image_array, 'mobilenet')
mobilenetv2_test_image_preprocessed = image_preprocess(mobilenetv2_test_image_array, 'mobilenet_v2')
inceptionv3_test_image_preprocessed = image_preprocess(inceptionv3_test_image_array, 'inception_v3')

print('image preprocessing completed!\n')


def save_model(model, saved_model_dir):
    model = models_detail[model]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    model.save(saved_model_dir, include_optimizer=False, save_format='tf')


#
# print('\nsaving and loading models...')
#
loaded_models = {}
#
# for model_name in models_to_load:
#     model_names = models_detail.keys()
#     if model_name in model_names:
#         model_path = f'{model_name}_saved_model'
#         if os.path.isdir(model_path) == False:
#             print('model save')
#             save_model(model_name, model_path)
#         loaded_models[model_name] = tf.keras.models.load_model(model_path)
#     else:
#         print(f'model names must be in {model_names}')
#         exit(1)
#
# print('saving and loading models completed!\n')


# Yolo v5
yolov5_image_path = './dataset/imagenet/imagenet_1000_raw/n02782093_1.JPEG'
weights = './model/yolov5/yolov5s.pt'
data = './model/yolov5/coco128.yaml'  # dataset.yaml path
imgsz = (640, 640)  # inference size (height, width)
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 1000  # maximum detections per image
vid_stride = 1,  # video frame-rate stride

yolo_model = DetectMultiBackend(weights, data=data)
stride, names, pt = yolo_model.stride, yolo_model.names, yolo_model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Dataloader
bs = 1  # batch_size
yolov5_dataset = LoadImages(yolov5_image_path, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

# Run inference
yolo_model.warmup(imgsz=(1 if pt or yolo_model.triton else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
path, im, im0s, vid_cap, s = yolov5_dataset[0]

# for path, im, im0s, vid_cap, s in yolov5_dataset:
#     with dt[0]:
#         im = torch.from_numpy(im).to(yolo_model.device)
#         im = im.half() if yolo_model.fp16 else im.float()  # uint8 to fp16/32
#         im /= 255  # 0 - 255 to 0.0 - 1.0
#         if len(im.shape) == 3:
#             im = im[None]  # expand for batch dim
#
#     # Inference
#     with dt[1]:
#         pred = yolo_model(im)
#
#     # NMS
#     with dt[2]:
#         pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
#
#     # Process predictions
#     for i, det in enumerate(pred):  # per image
#         seen += 1
#         p, im0, frame = path, im0s.copy(), getattr(yolov5_dataset, 'frame', 0)
#         p = Path(p)  # to Path
#
#     # Print time (inference-only)
#     LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
#
#
# # Print results
# t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
# LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


app = Flask(__name__)


@app.route('/mobilenet')
def mobilenetv1():
    inference_start_time = time.time()
    result = loaded_models['mobilenet'].predict(mobilenetv1_test_image_preprocessed)
    inference_end_time = time.time()

    inference_time = inference_end_time - inference_start_time

    return f'mobilenetv1 inference success\ninference time:{inference_time}\n'


@app.route('/mobilenet_v2')
def mobilenetv2():
    inference_start_time = time.time()
    result = loaded_models['mobilenet_v2'].predict(mobilenetv2_test_image_preprocessed)
    inference_end_time = time.time()

    inference_time = inference_end_time - inference_start_time

    # print(result)

    return f'mobilenetv2 inference success\ninference time:{inference_time}\n'


@app.route('/inception_v3')
def inceptionv3():
    inference_start_time = time.time()
    result = loaded_models['inception_v3'].predict(inceptionv3_test_image_preprocessed)
    inference_end_time = time.time()

    inference_time = inference_end_time - inference_start_time

    # print(result)

    return f'inceptionv3 inference success\ninference time:{inference_time}\n'


@app.route('/yolo_v5')
def yolov5():
    # for path, im, im0s, vid_cap, s in yolov5_dataset:
    #
    inference_start_time = time.time()
    result = yolo_model(im)
    inference_end_time = time.time()

    inference_time = inference_end_time - inference_start_time


#
# inference_start_time = time.time()
# yolov5_image_path = './dataset/imagenet/imagenet_1000_raw/n02782093_1.JPEG'
# weights = './model/yolov5/yolov5s.pt'
# data = './model/yolov5/coco128.yaml'  # dataset.yaml path
# imgsz = (640, 640)  # inference size (height, width)
# conf_thres = 0.25  # confidence threshold
# iou_thres = 0.45  # NMS IOU threshold
# max_det = 1000  # maximum detections per image
# vid_stride = 1,  # video frame-rate stride
#
# yolo_model = DetectMultiBackend(weights, data=data)
# stride, names, pt = yolo_model.stride, yolo_model.names, yolo_model.pt
# imgsz = check_img_size(imgsz, s=stride)  # check image size
#
# # Dataloader
# bs = 1  # batch_size
# yolov5_dataset = LoadImages(yolov5_image_path, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#
# # Run inference
# yolo_model.warmup(imgsz=(1 if pt or yolo_model.triton else bs, 3, *imgsz))  # warmup
# seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
# for path, im, im0s, vid_cap, s in yolov5_dataset:
#     with dt[0]:
#         im = torch.from_numpy(im).to(yolo_model.device)
#         im = im.half() if yolo_model.fp16 else im.float()  # uint8 to fp16/32
#         im /= 255  # 0 - 255 to 0.0 - 1.0
#         if len(im.shape) == 3:
#             im = im[None]  # expand for batch dim
#
#     # Inference
#     with dt[1]:
#         pred = yolo_model(im)
#
#     # NMS
#     with dt[2]:
#         pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
#
#     # Process predictions
#     for i, det in enumerate(pred):  # per image
#         seen += 1
#         p, im0, frame = path, im0s.copy(), getattr(yolov5_dataset, 'frame', 0)
#         p = Path(p)  # to Path
#
#     # Print time (inference-only)
#     LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
#
# # Print results
# t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
# LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
# inference_end_time = time.time()
#
# inference_time = inference_end_time - inference_start_time
    return f'yolov5 inference success\ninference time:{inference_time}\n'

app.run(host=hostname, port=port, threaded=False)
