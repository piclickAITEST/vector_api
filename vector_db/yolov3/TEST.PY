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
        # img.shpae: [3,416,416), img0.shape:[h, w, 3]
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

            ratio = kernel_size / img_size
            resized_det = det[:, :4] * ratio

            im0shape = img0.shape
            feature_box = np.asarray(resized_det.detach().cpu().numpy(), dtype=np.int32)

            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0shape).round()  # originial

            for (*xyxy, conf, cls), fb in zip(det, feature_box):
                if conf < 0.5: continue
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

