import tensorflow as tf
from tensorflow.keras.applications import inception_v3

Inceptionv3_model = tf.keras.applications.InceptionV3(weights='imagenet')


def representative_data_gen():
  batch_size=128
  dataset_list = tf.data.Dataset.list_files('../dataset/imagenet/imagenet_1000_raw/*.JPEG')
  dataset_list.shuffle(buffer_size=10000).batch(batch_size=batch_size)
  for i in enumerate(dataset_list.take(batch_size)):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [299, 299])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]



converter = tf.lite.TFLiteConverter.from_keras_model(Inceptionv3_model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()


with open('batch.tflite', 'wb') as f:
  f.write(tflite_model)
