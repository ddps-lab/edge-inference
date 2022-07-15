import wget

tfrecord_imagenet_1000 = 'https://jungae-imagenet-dataset.s3.amazonaws.com/imagenet_1000'
wget.download(tfrecord_imagenet_1000)

raw_imagenet_1000 = 'https://jungae-imagenet-dataset.s3.amazonaws.com/imagenet_1000_raw.zip'
wget.download(raw_imagenet_1000)
