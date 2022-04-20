### Image Classification

- ImageNet dataset 1000, 50000 download
    
    ```bash
    python3 ./dataset/ImageNet/val_dataset.py
    ```
    
- Image Classification model inference (MobilNet V1)
    
    ```bash
    python3 ic_inference.py --batchsize=1 --model=mobilenet --case=tf --quantization=FP32 --engines=1 --img_size=224
    ```
    
- Image Classification model inference (MobilNet V2)
    
    ```bash
    python3 ic_inference.py --batchsize=1 --model=mobilenet_v2 --case=tf --quantization=FP32 --engines=1 --img_size=224
    ```
    
- Image Classification model inference (Inception V3)
    
    ```bash
    python3 ic_inference.py --batchsize=1 --model=inception_v3 --case=tf --quantization=FP32 --engines=1 --img_size=299
    ```

### Object Detection


- COCO 2017 dataset 5000 download
    
    ```bash
    python3 ./dataset/COCO_2017/val_dataset.py
    unzip -q coco2017val.zip -d ./datasets && rm coco2017val.zip
    ```

- Object Detection model inference (YOLO V5)

    ```bash
    python3 ob_inference.py --weights ./CNN/model/yolo_v5_model/yolov5s_saved_model --data ./CNN/dataset/COCO_2017/coco.yaml --img 640 --iou 0.65 --half --task val
    ```
