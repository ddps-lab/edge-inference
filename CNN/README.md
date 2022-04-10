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


- COCO 2017 dataset 50000 download
    
    ```bash
    python3 ./dataset/COCO 2017/val_dataset.py
    ```
