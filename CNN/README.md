### Image Classification

- tfrecord ImageNet dataset 1000, ImageNet raw dataset 1000 download
    
    ```bash
    python3 ./dataset/imagenet/val_dataset.py
    mv ./imagenet_1000 ./dataset/imagenet/
    unzip -q imagenet_1000_raw.zip -d ./dataset/imagenet && rm imagenet_1000_raw.zip
    ```
    
- Image Classification model inference (MobilNet V1 - using tfrecord ImageNet dataset 1000)
    
    ```bash
    python3 ic_inference.py --batchsize=1 --model=mobilenet --case=tf --quantization=FP32 --engines=1 --img_size=224
    ```
    
- Image Classification model inference (MobilNet V2 - using tfrecord ImageNet dataset 1000)
    
    ```bash
    python3 ic_inference.py --batchsize=1 --model=mobilenet_v2 --case=tf --quantization=FP32 --engines=1 --img_size=224
    ```
    
- Image Classification model inference (Inception V3 - using tfrecord ImageNet dataset 1000)
    
    ```bash
    python3 ic_inference.py --batchsize=1 --model=inception_v3 --case=tf --quantization=FP32 --engines=1 --img_size=299
    ```

- Image Classification edgetpu tflite model inference (MobileNet v1 - using ImageNet raw dataset 1000)

    ```bash 
    python3 ic_inference_edgetpu.py --model ./model/mobilenet_v1_edgetpu_tflite/tf2_mobilenet_v1_1.0_224_ptq_edgetpu.tflite  --labels ./dataset/imagenet/imagenet_metadata.txt
    ```

- Image Classification edgetpu tflite model inference (MobileNet v2 - using ImageNet raw dataset 1000)

    ```bash 
    python3 ic_inference_edgetpu.py --model ./model/mobilenet_v2_edgetpu_tflite/tf2_mobilenet_v2_1.0_224_ptq_edgetpu.tflite  --labels ./dataset/imagenet/imagenet_metadata.txt
    ```

- Image Classification edgetpu tflite model inference (Inception v3 - using ImageNet raw dataset 1000)

    ```bash
    python3 ic_inference_edgetpu.py --model ./model/inception_v3_edgetpu_tflite/inceptionv3_edgetpu.tflite  --labels ./dataset/imagenet/imagenet_metadata.txt
    ```


### Object Detection (image)

- COCO 2017 dataset 5000 download
    
    ```bash
    python3 ./dataset/coco_2017/val_dataset.py
    unzip -q coco2017val.zip -d ./model/yolo_v5/datasets && rm coco2017val.zip
    ```

- Object Detection model inference (YOLO V5 - Nvidia Jetson & Coral TPU)

    ```bash
    python3 od_image_inference.py --weights ./model/yolo_v5/yolov5s_saved_model_FP32 --data ./model/yolo_v5/coco.yaml --batch-size 1
    ```
    ```bash
    python3 od_image_inference.py --weights ./model/yolo_v5_edgetpu_tflite/yolov5s-int8_edgetpu.tflite --data ./model/yolo_v5/coco.yaml --batch-size 1
    ```
    
### Object Detection (video)

- Object Detection model inference (YOLO V5 - Nvidia Jetson & Coral TPU)

    ```bash
    python3 od_video_inference.py --weights ./model/yolo_v5/yolov5s_saved_model_FP32  --source ./dataset/video/road.mp4 --data ./model/yolo_v5/coco.yaml
    ```
    ```bash
    python3 od_video_inference.py --weights ./model/yolo_v5_edgetpu_tflite/yolov5s-int8_edgetpu.tflite  --source ./dataset/video/road.mp4 --data ./model/yolo_v5/coco.yaml
    ```
