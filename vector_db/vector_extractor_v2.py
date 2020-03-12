import cv2
import time
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from sklearn.random_projection import SparseRandomProjection

from .yolov3.models import *
from .yolov3.pirs_utils_v2 import *
from torch.utils.data import DataLoader

try:
    from apex import amp
    mixed_precision = True
except:
    mixed_precision = False

img_size = 416

cfg = '/vector_api/vector_db/yolov3/cfg/fashion/fashion_c23.cfg'
weights = '/mnt/piclick/piclick.ai/weights/best.pt'
names_path = '/vector_api/vector_db/yolov3/data/fashion/fashion_c23.names'

device = torch_utils.select_device('', apex=mixed_precision)
model = Darknet(cfg, arc='default').to(device).eval()
model.load_state_dict(torch.load(weights, map_location=device)['model'])


class Yolov3:
    def vector_extraction_batch(self, data, batch_size = 1, rp=False, pooling='max'):
        img_res = {}
        list_names = []

        names = load_classes(names_path)

        for i in range(len(names)):
            list_names.append([])

        torch.cuda.empty_cache()
        dataset = LoadImages(data, img_size, batch_size=batch_size)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

        # [batch, 3, 416, 416], [path1, path2], [[h,w,3],[h,w,3]]
        for batch_i, (imgs, paths, shapes) in enumerate(tqdm(dataloader)):
            batch_time = time.time()
            torch.cuda.empty_cache()

            with torch.no_grad():
                imgs = imgs.to(device).float() / 255.0
                _, _, height, width = imgs.shape

                layerResult = LayerResult(model.module_list, 80)
                pred = model(imgs)[0]

                layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
                kernel_size = layerResult_tensor.shape[1]
                LayerResult.unregister_forward_hook(layerResult)

                # box info
                pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                try:
                    if det is not None and len(det):
                        ratio = kernel_size / img_size
                        resized_det = det[:, :4] * ratio

                        im0shape = shapes[i]  # original shape
                        feature_box = np.asarray(resized_det.detach().cpu().numpy(), dtype=np.int32)

                        det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], im0shape).round()  # originial

                        for (*xyxy, conf, cls), fb in zip(det, feature_box):
                            if conf < 0.7: continue
                            feature_result = layerResult_tensor[i, fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                            xyxy = [int(x) for x in xyxy]
                            xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                            xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                            class_data = {
                                'raw_box': xyxy,
                                'feature_vector': max_pooling_tensor(
                                    feature_result) if pooling == 'max' else average_pooling_tensor(feature_result),
                                'img_path': paths[i]
                            }

                            list_names[int(cls)].append(class_data)
                except:
                    continue

            batch_end = time.time() - batch_time
            print(" Inference time for a image : {}".format(batch_end / batch_size))
            print(" Inference time for batch image : {}".format(batch_end))

            if rp:
                print("Reduce Vector Dimension 1024 to 512 ..")
                time_rd = time.time()
                rng = np.random.RandomState(42)
                transformer = SparseRandomProjection(n_components=512, random_state=rng)

                for i, ln in enumerate(list_names):
                    if list_names[i] != []:
                        ln_df = pd.DataFrame(pd.DataFrame.from_dict(ln)['feature_vector'].tolist())
                        ln_df_x = ln_df.loc[:, :].values  # numpy arrays
                        X_new = transformer.fit_transform(ln_df_x)
                        # 재저장
                        for j in range(len(ln)):
                            ln[j]['feature_vector'] = X_new[j]
                print(".. ", time.time() - time_rd)

            for i in range(len(names)):
                if list_names[int(i)] != []:
                    img_res[names[int(i)]] = list_names[int(i)]

        return img_res


    def vector_extraction_service(self, base_img, pooling='max'):
        img_res = {}
        list_names = []

        names = load_classes(names_path)

        for i in range(len(names)):
            list_names.append([])

        batch_time = time.time()
        torch.cuda.empty_cache()

        with torch.no_grad():
            img_bytes = base64.b64decode(base_img)
            file_bytes = np.asarray(bytearray(BytesIO(img_bytes).read()), dtype=np.uint8)
            decode_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            img, img0 = transfer_b64(decode_img, mode='square')  # auto, square, rect, scaleFill / default : square
            #img.shpae: [3,416,416), img0.shape:[h, w, 3]
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            layerResult = LayerResult(model.module_list, 80)

            pred = model(img)[0]

            layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
            kernel_size = layerResult_tensor.shape[1]

            LayerResult.unregister_forward_hook(layerResult)

            # box info
            pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                padded_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype=np.float32)

                ratio = kernel_size / img_size
                resized_det = det[:, :4] * ratio

                im0shape = img0.shape
                feature_box = np.asarray(resized_det.detach().cpu().numpy(), dtype=np.int32)

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()  # originial

                for (*xyxy, conf, cls), fb in zip(det, feature_box):
                    #if conf < 0.5: continue
                    feature_result = layerResult_tensor[i, fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                    xyxy = [int(x) for x in xyxy]
                    xyxy[0], xyxy[2] = xyxy[0] / im0shape[1], xyxy[2] / im0shape[1]
                    xyxy[1], xyxy[3] = xyxy[1] / im0shape[0], xyxy[3] / im0shape[0]

                    class_data = {
                        'raw_box': xyxy,
                        'feature_vector': max_pooling_tensor(
                            feature_result) if pooling == 'max' else average_pooling_tensor(feature_result),
                    }

                    list_names[int(cls)].append(class_data)

        for i in range(len(names)):
            if list_names[int(i)] != []:
                img_res[names[int(i)]] = list_names[int(i)]

        batch_end = time.time() - batch_time
        print("Inference time for a image : {}".format(batch_end))

        return img_res
