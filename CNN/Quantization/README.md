
### Model Quantization 

- Image Classification model tflite quantization convert (Mobilenet V2, Inception V3)
    ```bash
    python3 ./MobileNet V2_tflite_convert.py
    python3 ./Inception V3_tflite_convert.py
    ```
- Image Classification model edgetpu-tflite quantization convert (edgetpu-compiler v15.0, edgetpu-runtime v15.0)
    ```bash
    curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/compiler.zip
    unzip compiler.zip
    ./compiler/edgetpu_compiler [tflite model]
    ```
