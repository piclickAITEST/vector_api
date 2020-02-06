import numpy as np
import cv2
import torch
import skimage.measure

from .utils.utils import *

class LayerResult:
    def __init__(self, payers, layer_index):
        self.hook = payers[layer_index].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu().data.numpy()

    def unregister_forward_hook(self):
        self.hook.remove()

def letterbox(img, new_shape=416, color=(128,128,128), mode='square'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
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
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratiow, ratioh, dw, dh


def transfer(path, mode):
    img_size = 416
    img0 = cv2.imread(path)
    img = letterbox(img0, new_shape=img_size, mode=mode)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    return img, img0

def max_pooling(result_np, kernel_size):
    size = int(np.ceil(result_np.shape[0] / kernel_size))
    dim = result_np.shape[-1]
    for i in range(dim):
        a = result_np[:, :, i]
        if i == 0:
            reduced_a = np.zeros([size, size, dim], dtype=np.float32)

        reduced_a[:, :, i] = skimage.measure.block_reduce(a, (kernel_size, kernel_size), np.max)

    return reduced_a, dim

def average_pooling(result_np, kernel_size):
    size = int(np.ceil(result_np.shape[0] / kernel_size))
    dim = result_np.shape[-1]
    for i in range(dim):
        a = result_np[:, :, i]
        if i == 0:
            reduced_a = np.zeros([size, size, dim], dtype=np.float32)

        reduced_a[:, :, i] = skimage.measure.block_reduce(a, (kernel_size, kernel_size), np.average)

    return reduced_a, dim

