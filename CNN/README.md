### Image Classification


- Image Classification model inference (Nvidia Jetson - MobileNet V1 - using tfrecord ImageNet dataset 1000)
    
    ```bash
    python3 image_classification_inference.py --batchsize=1 --model=mobilenet --case=tf --quantization=FP32 --engines=1 --img_size=224
    ```
    
- Image Classification model inference (Nvidia Jetson - MobileNet V2 - using tfrecord ImageNet dataset 1000)
    
    ```bash
    python3 image_classification_inference.py --batchsize=1 --model=mobilenet_v2 --case=tf --quantization=FP32 --engines=1 --img_size=224
    ```
    
- Image Classification model inference (Nvidia Jetson - Inception V3 - using tfrecord ImageNet dataset 1000)
    
    ```bash
    python3 image_classification_inference.py --batchsize=1 --model=inception_v3 --case=tf --quantization=FP32 --engines=1 --img_size=299
    ```

- Image Classification edgetpu tflite model inference (Coral - MobileNet v1 - using ImageNet raw dataset 1000)

    ```bash 
    python3 image_classification_edgetpu_inference.py --model ./model/mobilenet_v1_edgetpu_tflite/tf2_mobilenet_v1_1.0_224_ptq_edgetpu.tflite  --labels ./dataset/imagenet/imagenet_metadata.txt
    ```

- Image Classification edgetpu tflite model inference (Coral - MobileNet v2 - using ImageNet raw dataset 1000)

    ```bash 
    python3 image_classification_edgetpu_inference.py --model ./model/mobilenet_v2_edgetpu_tflite/tf2_mobilenet_v2_1.0_224_ptq_edgetpu.tflite  --labels ./dataset/imagenet/imagenet_metadata.txt
    ```

- Image Classification edgetpu tflite model inference (Coral - Inception v3 - using ImageNet raw dataset 1000)

    ```bash
    python3 image_classification_edgetpu_inference.py --model ./model/inception_v3_edgetpu_tflite/inception_v3_299_quant_edgetpu.tflite  --labels ./dataset/imagenet/imagenet_metadata.txt
    ```


### Object Detection (image)

- Object Detection model inference (Nvidia Jetson & Coral TPU - YOLO V5 - COCO dataset 5000)

    ```bash
    python3 object_detection_image_inference.py--weights ./model/yolo_v5/yolov5s_saved_model_FP32 --data ./model/yolo_v5/coco.yaml --batch-size 1
    ```
    ```bash
    python3 object_detection_image_inference.py --weights ./model/yolo_v5_edgetpu_tflite/yolov5s-int8_edgetpu.tflite --data ./model/yolo_v5/coco.yaml --batch-size 1
    ```
    
### Object Detection (video)

- Object Detection model inference (Nvidia Jetson & Coral TPU - YOLO V5 - road video 20s/38frame)

    ```bash
    python3 object_detection_video_inference.py --weights ./model/yolo_v5/yolov5s_saved_model_FP32  --source ./dataset/video/road.mp4 --data ./model/yolo_v5/coco.yaml
    ```
    ```bash
    python3 object_detection_video_inference.py --weights ./model/yolo_v5_edgetpu_tflite/yolov5s-int8_edgetpu.tflite  --source ./dataset/video/road.mp4 --data ./model/yolo_v5/coco.yaml
    ```

- Object Detection tflite quantization model inference (Nvidia Jetson & Coral TPU - YOLO V5 - road video 20s/38frame)

    ```bash
    python3 object_detection_video_inference.py --weights ./model/yolo_v5_quantization_tflite/yolov5s-fp16.tflite  --source ./dataset/video/road.mp4 --data ./model/yolo_v5/coco.yaml
    ```
    ```bash
    python3 object_detection_video_inference.py --weights ./model/yolo_v5_quantization_tflite/yolov5s-int8.tflite  --source ./dataset/video/road.mp4 --data ./model/yolo_v5/coco.yaml
    ```
