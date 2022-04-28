import argparse
import time
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm.auto import tqdm

from model.yolo_v5.models.common import DetectMultiBackend
from model.yolo_v5.utils.callbacks import Callbacks
from model.yolo_v5.utils.datasets import create_dataloader
from model.yolo_v5.utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from model.yolo_v5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from model.yolo_v5.utils.plots import output_to_target, plot_images, plot_val_study
from model.yolo_v5.utils.torch_utils import select_device, time_sync


def process_batch(detections, labels, iouv):

    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=1,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.5,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    device = select_device(device, batch_size=batch_size)


    # Load model
    model_load_time = time.time()
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    model_load_time = time.time() - model_load_time

    # model, data, device Configure
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    device = model.device
    #batch_size = 1  # export.py models default to batch-size 1
    data = check_dataset(data)  # check
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Data load
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    pad = 0.5
    rect = pt  # square inference for benchmarks
    task = 'val'  # path to train/val/test images
    dataset_load_time = time.time()
    dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    dataset_load_time = time.time() - dataset_load_time
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    
    # model inference
    iftime = []
    iftime_start = time.time()
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # image inference
        iftime_avg_start = time.time()
        out, train_out = model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = []  
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

        iftime_avg_end = time.time() - iftime_avg_start
        iftime.append(iftime_avg_end)
    iftime = np.array(iftime)
    inference_time = time.time() - iftime_start


    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))


    # metric
    print('accuracy = ', mr)
    print('model_load_time =', model_load_time) 
    print('dataset_load_time =', dataset_load_time)
    print('inference_time =', inference_time)
    print('inference_time(avg) =',np.sum(iftime)/(len(iftime)*batch_size))
    print('IPS =', (len(iftime)*batch_size)/(model_load_time + dataset_load_time + inference_time))
    print('IPS(inf) =', (len(iftime)*batch_size)/np.sum(iftime))
    print(len(iftime))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./yolo_v5/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='./yolo_v5/yolov5s_saved_model', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML

    print_args(vars(opt))
    return opt


def main(opt):
    if opt.task == 'val':  # run normally
        run(**vars(opt))
    os.system('rm -rf ./*.png')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
