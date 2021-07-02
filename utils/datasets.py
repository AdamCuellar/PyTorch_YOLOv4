import glob
import math
import os
import shutil
import time
from pathlib import Path
from threading import Thread
import cv2
import random
import numpy as np
import torch
from tqdm import tqdm
from functools import partial

from utils.general import xyxy2xywh, torch_distributed_zero_first, plot_images

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']

def parsePathFile(topPth, path):
    """ Parses *.txt file with image paths"""
    if os.path.isabs(path):
        path = os.path.join(topPth, path)

    path = str(Path(path))
    assert os.path.isfile(path), "File not found {}".format(path)

    with open(path, "r") as f:
        lines = f.readlines()

    imgPaths = [line.strip() for line in lines]
    return imgPaths

def parseDataFile(path):
    """ Parses *.data file"""
    path = str(Path(path))
    assert os.path.isfile(path), "File not found {}".format(path)

    with open(path, "r") as f:
        lines = f.readlines()

    info = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        info[key.strip()] = val.strip()

    return info

def readTruth(file):
    l = []
    if os.path.isfile(file):
        with open(file, 'r') as f:
            l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
    if len(l) == 0:
        l = np.zeros((0, 5), dtype=np.float32)

    return l

def create_dataloader(dataFile, netParams, batch_size, imgShape=None, valid=False, test=False, local_rank=-1, world_size=1):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(local_rank):
        dataset = LoadData(dataFile, netParams, imgShape=imgShape, valid=valid, test=test)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if local_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             sampler=train_sampler,
                                             pin_memory=True,
                                             collate_fn=LoadData.collate_fn)
    return dataloader, dataset

def load_image(imgPath, img_size, resize=True):
    img = cv2.imread(imgPath)
    imgH, imgW = img.shape[:2]
    if resize and (imgH != img_size[0] and imgW != img_size[1]):
        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    return img, (imgH, imgW), img.shape[:2]

def load_mosaic(index, imgPaths, labels, img_size, aug=None):
    # loads images in a mosaic
    labels4 = []
    s = img_size
    xc, yc = s[1], s[0]
    indices = [index] + [random.randint(0, len(labels) - 1) for _ in range(3)]  # 3 additional image indicess
    for i, index in enumerate(indices):
        doResize = random.uniform(0, 1) > 0.25

        # Load image
        img, _, (h, w) = load_image(imgPaths[index], img_size, resize=doResize)

        # Labels
        x = labels[index].copy()

        if aug is not None:
            img, x = aug(img, x)

        currLabels = x.copy()

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s[1] * 2, s[0] * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[0] * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[1] * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[0] * 2), min(s[1] * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        if x.size > 0:  # Normalized xywh to pixel xyxy format
            currLabels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            currLabels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            currLabels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            currLabels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(currLabels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:3], 0, 2 * s[1], out=labels4[:, 1:3])  # use with random_affine
        np.clip(labels4[:, 3:], 0, 2 * s[0], out=labels4[:, 3:])

    return img4, labels4

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.asarray([random.uniform(-1, 1) for _ in range(3)]) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

def random_flip(img, labels, lr=True, ud=False):
    nL = len(labels)
    # random left-right flip
    if lr and random.uniform(0, 1) < 0.5:
        img = np.fliplr(img)
        if nL:
            labels[:, 1] = 1 - labels[:, 1]

    # random up-down flip
    if ud and random.uniform(0, 1) < 0.5:
        img = np.flipud(img)
        if nL:
            labels[:, 2] = 1 - labels[:, 2]

    return img, labels

def labels_to_relative_xyxy(x, h, w):
    copyX = x.copy()
    if len(x):
        x[:, 1] = w * (copyX[:, 1] - copyX[:, 3] / 2)
        x[:, 2] = h * (copyX[:, 2] - copyX[:, 4] / 2)
        x[:, 3] = w * (copyX[:, 1] + copyX[:, 3] / 2)
        x[:, 4] = h * (copyX[:, 2] + copyX[:, 4] / 2)
    return x

def labels_to_absolute_xywh(x, h, w):
    if len(x):
        x[:, 1:5] = xyxy2xywh(x[:, 1:5])  # convert xyxy to xywh
        x[:, [2, 4]] /= h  # normalized height 0-1
        x[:, [1, 3]] /= w  # normalized width 0-1
    return x

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

        # Transform label coordinates if we changed the image
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            if perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
            targets = targets[i]
            targets[:, 1:5] = xy[i]


    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

def aug_func(img, labels, hue, saturation, exposure, resize):
    # TODO: add jitter?
    augment_hsv(img, hgain=hue, sgain=saturation, vgain=exposure)
    img, labels = random_flip(img, labels)
    labels = labels_to_relative_xyxy(labels, img.shape[0], img.shape[1])
    img, labels = random_perspective(img, labels, degrees=0, translate=0, scale=resize, shear=0, perspective=0.0, border=(0, 0))
    labels = labels_to_absolute_xywh(labels, img.shape[0], img.shape[1])
    return img, labels

class LoadData(torch.utils.data.Dataset):
    def __init__(self, dataFile, netParams, imgShape=None, valid=False, test=False):
        self.netParams = netParams
        self.imgShape = (self.netParams["height"], self.netParams["width"]) if imgShape is None else imgShape
        self.imgShape = np.asarray(self.imgShape)
        self.aug = not valid and not test # TODO: maybe add test-time augmentation?
        # get location of data file
        topPth = os.path.abspath(dataFile).replace(dataFile, "")
        dataInfo = parseDataFile(dataFile)
        currSet = dataInfo["valid"] if valid or test else dataInfo["train"]
        imgPaths = parsePathFile(topPth, currSet)
        self.imgPaths, self.labels = [], []

        # verify dataset has images/truths
        numMissing, numFound, numEmpty, numDup = 0,0,0,0
        pbar = tqdm(imgPaths)
        for imgPath in pbar:
            fName, ext = os.path.splitext(imgPath)
            txtPath = imgPath.replace(ext, ".txt")

            if os.path.exists(imgPath):
                self.imgPaths.append(imgPath)
                boxes = readTruth(txtPath)
                self.labels.append(boxes)

                if boxes.shape[0]:
                    assert boxes.shape[1] == 5, '> 5 label columns: %s' % txtPath
                    assert (boxes >= 0).all(), 'negative labels: %s' % txtPath
                    assert (boxes[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % txtPath
                    if np.unique(boxes, axis=0).shape[0] < boxes.shape[0]:  # duplicate rows
                        numDup += 1
                    numFound += 1
                else:
                    numEmpty += 1
            else:
                numMissing += 1

            pbar.desc = "Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)" \
                        % (currSet, numFound, numMissing, numEmpty, numDup, len(imgPaths))

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        doMosaic = self.aug and "mosaic" in self.netParams and self.netParams["mosaic"]
        augmentFunc = None
        if self.aug:
            augmentFunc = partial(aug_func,
                                  hue=self.netParams["hue"],
                                  saturation=1-self.netParams["saturation"],
                                  exposure=1-self.netParams["exposure"],
                                  resize=self.netParams["resize"])

        if doMosaic:
            img, labels = load_mosaic(index, self.imgPaths, self.labels, self.imgShape, aug=augmentFunc)
        else:
            img, _, (h, w) = load_image(self.imgPaths[index], self.imgShape, resize=False)
            labels = self.labels[index].copy()

            if self.aug:
                img, labels = augmentFunc(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        shape = img.shape[:-1]
        detImg = cv2.resize(img, self.imgShape[::-1], interpolation=cv2.INTER_LINEAR)
        detImg = detImg[:, :, ::-1].transpose(2, 0, 1)
        detImg = np.ascontiguousarray(detImg)

        # plot_images(np.expand_dims(detImg, 0), labels_out, fname="images_{}.jpg".format(index))
        return torch.from_numpy(detImg), labels_out, self.imgPaths[index], shape

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes