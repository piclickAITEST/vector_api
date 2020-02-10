import numpy as np
import cv2
import torch
import skimage.measure

import torch.nn.functional as F
from torch.utils.data import Dataset
from .utils.utils import *

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

class LayerResult:
    def __init__(self, payers, layer_index):
        self.hook = payers[layer_index].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features_np = output.cpu().data.numpy()
        self.features = output

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

def max_pooling_tensor(result_tensor):

    result_tensor = result_tensor.reshape(-1, 1, 1024).squeeze(1)
    max_value, _ = torch.max(result_tensor,0)
    max_value = max_value.detach().cpu().numpy()

    return max_value

def average_pooling_tensor(result_tensor):

    result_tensor = result_tensor.reshape(-1, 1, 1024).squeeze(1)
    average_value, _ = torch.mean(result_tensor,0)

    return average_value.detach().cpu().numpy()

def max_pooling_np(result_np, kernel_size):
    size = int(np.ceil(result_np.shape[0] / kernel_size))
    dim = result_np.shape[-1]
    for i in range(dim):
        a = result_np[:, :, i]
        if i == 0:
            reduced_a = np.zeros([size, size, dim], dtype=np.float32)

        reduced_a[:, :, i] = skimage.measure.block_reduce(a, (kernel_size, kernel_size), np.max)

    return reduced_a, dim

def average_pooling_np(result_np, kernel_size):
    size = int(np.ceil(result_np.shape[0] / kernel_size))
    dim = result_np.shape[-1]
    for i in range(dim):
        a = result_np[:, :, i]
        if i == 0:
            reduced_a = np.zeros([size, size, dim], dtype=np.float32)

        reduced_a[:, :, i] = skimage.measure.block_reduce(a, (kernel_size, kernel_size), np.average)

    return reduced_a, dim

def transfer(path, mode):
    img_size = 416
    img0 = cv2.imread(path)
    img = letterbox(img0, new_shape=img_size, mode=mode)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    return img, img0

class LoadImages(Dataset):
    def __init__(self, path, img_size=416, batch_size=16):

        #path = glob.glob(path)
        self.img_files = [x for x in path if "." + x.split('.')[-1].lower() in img_formats]

        n = len(self.img_files)
        assert n > 0, 'No images found in %s' % (path)
        self.imgs = [None] * n
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index

        self.n = n
        print("Total images {}".format(self.n))
        self.augment = False
        self.batch = bi  # batch index of image
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]

        # Load image
        img = load_image(self, index)

        # Letterbox
        h, w = img.shape[:2]
        img, ratiow, ratioh, _, _ = letterbox(img, new_shape=416, color=(128,128,128), mode='square')

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), img_path, ((h, w), (ratiow, ratioh))

    @staticmethod
    def collate_fn(batch):
        img, path, shapes = list(zip(*batch))  # transposed
        return torch.stack(img, 0), path, shapes

def load_image(self, index):
    # loads 1 image from dataset
    img = self.imgs[index]
    if img is None:
        img_path = self.img_files[index]
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'Image Not Found ' + img_path
        r = self.img_size / max(img.shape)  # resize image to img_size
        if self.augment and (r != 1):  # always resize down, only resize up if training with augmentation
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # _LINEAR fastest
    return img