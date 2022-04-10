import wget

imagenet_1000 = 'https://jungae-imagenet-dataset.s3.amazonaws.com/imagenet_1000'
wget.download(imagenet_1000)

imagenet_50000 = 'https://jungae-imagenet-dataset.s3.amazonaws.com/imagenet_50000'
wget.download(imagenet_50000)
