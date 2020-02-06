from .yolov3.models import *
from .yolov3.utils.utils import *
from .yolov3.pirs_utils import *


class Yolov3():
    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
        print("Apex Loaded")
    except:
        mixed_precision = False  # not installed

    cfg = '/home/piclick/vector_api/vector_db/yolov3/cfg/fashion/fashion_c14.cfg'
    img_size = 416
    batch_size = 64
    weights = '/home/piclick/vector_api/vector_db/yolov3/weights/exp3_best_20200109.pt'
    device = torch_utils.select_device('', apex=mixed_precision, batch_size=batch_size)

    model = Darknet(cfg, arc='default').to(device).eval()

    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        print('no weight')


    def vector_extraction_with_folder(self, bulk_path):
        img_res = {}
        list_names = []

        names_path = '/home/piclick/vector_api/vector_db/yolov3/data/fashion/fashion_c14.names'
        names = load_classes(names_path)

        for i in range(len(names)):
            list_names.append([])

        for path in glob.glob(bulk_path + '*.*'):
            img, img0 = transfer(path, mode='square') # auto, square, rect, scaleFill / default : square
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            result1 = LayerResult(self.model.module_list, 80)
            pred = self.model(img)[0]

            result1_np = result1.features.squeeze(0).transpose([1, 2, 0])
            dim = result1_np.shape[-1]
            kernel_size = result1_np.shape[0]

            # Normalize
            for d in range(dim):
                dmax = result1_np[:, :, d].max()
                dmin = result1_np[:, :, d].min()
                result1_np[:, :, d] = (result1_np[:, :, 0] - dmin) / (dmax - dmin)

            # box info
            pred = non_max_suppression(pred, 0.5, 0.5)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):

                    raw_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype= np.float32)

                    ratio = kernel_size / self.img_size
                    det[:, :4] = det[:, :4] * ratio

                    feature_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype=np.int32)

                    for (*xyxy, conf, cls), fb, rb in zip(det, feature_box, raw_box):
                        if conf < 0.9: continue
                        feature_result = result1_np[fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                        class_data = {'path': path.split('/')[-1],
                                    'raw_box': rb,
                                    'feature_box' : fb,
                                    'feature_vector_max': max_pooling(feature_result, kernel_size)[0][0],
                                    'featrue_vector_average': average_pooling(feature_result, kernel_size)[0][0]}
                        list_names[int(cls)].append(class_data)

            for i in range(len(names)):
                if list_names[int(i)] != []:
                    img_res[names[int(i)]] = list_names[int(i)]

        return img_res

    def vector_extraction_with_img_path(self, img_path):
        img_res = {}
        list_names = []

        names_path = '/home/piclick/vector_api/vector_db/yolov3/data/fashion/fashion_c14.names'
        names = load_classes(names_path)

        for i in range(len(names)):
            list_names.append([])

        img, img0 = transfer(img_path, mode='square') # auto, square, rect, scaleFill / default : square
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        result1 = LayerResult(self.model.module_list, 80)
        pred = self.model(img)[0]

        result1_np = result1.features.squeeze(0).transpose([1, 2, 0])
        dim = result1_np.shape[-1]
        kernel_size = result1_np.shape[0]

        # Normalize
        for d in range(dim):
            dmax = result1_np[:, :, d].max()
            dmin = result1_np[:, :, d].min()
            result1_np[:, :, d] = (result1_np[:, :, 0] - dmin) / (dmax - dmin)

        # box info
        pred = non_max_suppression(pred, 0.5, 0.5)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                raw_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype= np.float32)

                ratio = kernel_size / self.img_size
                det[:, :4] = det[:, :4] * ratio

                feature_box = np.asarray(det[:, :4].detach().cpu().numpy(), dtype=np.int32)

                for (*xyxy, conf, cls), fb, rb in zip(det, feature_box, raw_box):
                    print(conf)
                    if conf < 0.5: continue
                    feature_result = result1_np[fb[0]:fb[2] + 1, fb[1]:fb[3] + 1]

                    class_data = {'path': img_path,
                                'raw_box': rb,
                                'feature_box' : fb,
                                'feature_vector_max': max_pooling(feature_result, kernel_size)[0][0],
                                'featrue_vector_average': average_pooling(feature_result, kernel_size)[0][0][0]}
                    list_names[int(cls)].append(class_data)

        for i in range(len(names)):
            if list_names[int(i)] != []:
                img_res[names[int(i)]] = list_names[int(i)]

        return img_res

'''
if __name__ == '__main__':
    print('init')
    #bulk_path = 'img_test/'
    #img_res = vector_extraction(bulk_path)
'''