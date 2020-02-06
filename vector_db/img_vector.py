import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchvision import models

import yolov3.test  # import test.py to get mAP after each epoch
from yolov3.models import *
from yolov3.utils.datasets import *
from yolov3.utils.utils import *

import matplotlib.pyplot as plt

from PIL import Image

import skimage.measure
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import cv2

class yolov3():
    cfg = 'yolov3/cfg/fashion/fashion_c14.cfg'
    data = 'yolov3/data/fashion/fashion_c14.data'
    img_size = 416
    epochs = 300
    batch_size = 128
    accumulate = 1
    weights = 'yolov3/weights/best.pt'
    device = torch_utils.select_device('',batch_size=batch_size)

    init_seeds()

    model = Darknet(cfg, arc='default').to(device).eval()
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    def transfer(self, path):
        img_size = 416
        img0 = cv2.imread(path)
        img = letterbox(img0, new_shape=img_size, mode='square')[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img, img0

    def letterbox(self, img, new_shape=416, color=(128, 128, 128), mode='square'):

        shape = img.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            ratio = float(new_shape) / max(shape)
        else:
            ratio = max(new_shape) / max(shape)  # ratio  = new / old
        ratiow, ratioh = ratio, ratio
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if mode is 'auto':  # minimum rectangle
            dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
            dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
        elif mode is 'square':  # square
            dw = (new_shape - new_unpad[0]) / 2  # width padding
            dh = (new_shape - new_unpad[1]) / 2  # height padding
        elif mode is 'rect':  # square
            dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
            dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
        elif mode is 'scaleFill':
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape, new_shape)
            ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad,
                             interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratiow, ratioh, dw, dh

    def maxpooling(self, result_np, kernel_size):

        size = int(np.ceil(result_np.shape[0] / kernel_size))
        dim = result_np.shape[-1]
        for i in range(dim):
            a = result_np[:, :, i]
            if i == 0:
                reduced_a = np.zeros([size, size, dim], dtype=np.float32)

            reduced_a[:, :, i] = skimage.measure.block_reduce(a, (kernel_size, kernel_size), np.max)  # MAX POOLING

        return reduced_a, dim

    def forward(self, path):
        img, img0 = transfer(path)
        img = torch.from_numpy(img).to(device)

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        vector = LayerResult(model.module_list, 80)
        pred = model(img)[0]
        vector_np = vector.features.squeeze(0).transpose([1, 2, 0])

        dim = vector_np.shape[-1]
        # Normalize
        for d in range(dim):
            dmax = vector_np[:, :, d].max()
            dmin = vector_np[:, :, d].min()
            vector_np[:, :, d] = (vector_np[:, :, 0] - dmin) / (dmax - dmin)

        pred = non_max_suppression(pred, 0.5, 0.5)
        names = load_classes("data/fashion/fashion_c14.names")
        feature_result_list = []

        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size

                # print(scale_coords(img.shape[2:], det[:, :4], img0.shape).round()) # input - raw
                print(det)

                ratio = 13 / 416
                det[:, :4] = det[:, :4] * ratio

                feature_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype=np.int32)

                for (*xyxy, conf, cls), fb in zip(det, feature_box):
                    if conf < 0.7: continue
                    print(names[int(cls)])
                    feature_result = vector_np[fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]
                    # print(feature_result.shape)
                    data = {'category': names[int(cls)],
                            'feature_result': list(self.maxpooling(feature_result,13)[0][0]),
                            'origin_xyxy': }

                    feature_result_list.append(data)

        img_res[path.split('/')[-1]] = feature_result_list


        ##pooling

        return new_vector
