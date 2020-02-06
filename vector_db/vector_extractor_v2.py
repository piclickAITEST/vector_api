from .yolov3.models import *
from .yolov3.utils.utils import *
from .yolov3.pirs_utils_v2 import *

import time
from torch.utils.data import DataLoader

class Yolov3():
    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
        print("Apex Loaded")
    except:
        mixed_precision = False  # not installed

    cfg = '/home/piclick/vector_api/vector_db/yolov3/cfg/fashion/fashion_c14.cfg'
    names_path = '/home/piclick/vector_api/vector_db/yolov3/data/fashion/fashion_c14.names'
    img_size = 416
    batch_size = 64
    weights = '/home/piclick/vector_api/vector_db/yolov3/weights/exp3_best_20200109.pt'
    device = torch_utils.select_device('', apex=mixed_precision, batch_size=batch_size)

    model = Darknet(cfg, arc='default').to(device).eval()

    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        print('no weight')

    def vector_extraction_batch(bulk_path, batch_size, pooling = 'max'):
        img_res = {}
        list_names = []

        names = load_classes(self.names_path)

        for i in range(len(names)):
            list_names.append([])

        dataset = LoadImages(bulk_path, img_size, batch_size=batch_size)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

        for batch_i, (imgs, paths, shapes) in enumerate(tqdm(dataloader)):
            batch_time = time.time()

            torch.cuda.empty_cache()
            with torch.no_grad():
                imgs = imgs.to(device).float() / 255.0
                _, _, height, width = imgs.shape

                layerResult = LayerResult(model.module_list, 80)

                pred = model(imgs)[0]

                layerResult_tensor = layerResult.features.permute([0,2,3,1])
                kernel_size = layerResult_tensor.shape[1]

                # box info
                pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):

                    raw_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype= np.float32)

                    ratio = kernel_size / img_size
                    det[:, :4] = det[:, :4] * ratio

                    feature_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype=np.int32)

                    for (*xyxy, conf, cls), fb, rb in zip(det, feature_box, raw_box):
                        if conf < 0.9: continue
                        feature_result = layerResult_tensor[i,fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                        class_data = {
                                    'raw_box': rb,
                                    'feature_vector': max_pooling_tensor(feature_result) if pooling == 'max' else average_pooling_tensor(feature_result),
                                    }


                        list_names[int(cls)].append(class_data)

            for i in range(len(names)):
                if list_names[int(i)] != []:
                    img_res[names[int(i)]] = list_names[int(i)]

            batch_end = time.time() - batch_time
            print(" Inference time for a image : {}".format(batch_end / batch_size))

        return img_res

    def vector_extraction_one_img(self, img_path, pooling='max'):
        img_res = {}
        list_names = []

        names = load_classes(self.names_path)

        for i in range(len(names)):
            list_names.append([])

        if not "." + img_path.split('.')[-1].lower() in img_formats:
            return -1

        batch_time = time.time()
        torch.cuda.empty_cache()
        with torch.no_grad():
            img, img0 = transfer(img_path, mode='square')  # auto, square, rect, scaleFill / default : square

            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            layerResult = LayerResult(self.model.module_list, 80)

            pred = self.model(img)[0]

            layerResult_tensor = layerResult.features.permute([0, 2, 3, 1])
            kernel_size = layerResult_tensor.shape[1]

            # box info
            pred = non_max_suppression(pred, conf_thres=0.5, nms_thres=0.5)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                raw_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype=np.float32)

                ratio = kernel_size / self.img_size
                det[:, :4] = det[:, :4] * ratio

                feature_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype=np.int32)

                for (*xyxy, conf, cls), fb, rb in zip(det, feature_box, raw_box):
                    if conf < 0.5: continue
                    feature_result = layerResult_tensor[i, fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                    class_data = {
                        'raw_box': rb,
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
