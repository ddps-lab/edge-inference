import wget

coco_2017 = 'ttps://ultralytics.com/assets/coco2017val.zip'
wget.download(coco_2017)
unzip -q coco2017val.zip -d ../datasets && rm coco2017val.zip
