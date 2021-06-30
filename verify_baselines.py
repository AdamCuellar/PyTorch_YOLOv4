import argparse
from sys import platform

from utils.model import *
from utils.datasets import *
from utils.general import *
import utils.torch_utils
import json
import torch
torch.set_printoptions(precision=8)

def detect(save_img=False):
    img_size = (opt.height, opt.width)
    out, source, weights, half, view_img, save_txt, save_json, save_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt, opt.save_json, opt.save_img

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        model.load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    dataset = LoadData(source, imgShape=img_size, test=True) # TODO: fix this

    # Get names and colors
    names = parse_names(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, img_size[0], img_size[1]), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    jdict = []
    coco91class = coco80_to_coco91_class()
    numOver = 0
    for path, img, im0s, vid_cap in dataset:
        print("########## Starting real image ################")
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        print("Time: {}".format(t2-t1))
        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   merge=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):

                # Append to pycocotools JSON dictionary
                if save_json:
                    det1 = det.clone()

                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(os.path.splitext(path.split("/")[-1])[0])
                    box = det1[:, :4].clone()  # xyxy
                    scale_coords(img.shape[2:], box, im0.shape, noLetter=True)  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    count = 0
                    for pr, b in zip(det.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                      'category_id': coco91class[int(pr[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(pr[4], 5)})
                        count += 1

                    if count > 100:
                        numOver += 1

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if True or save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)

    print("Number of images with detections over 100: {}".format(numOver))
    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)


    # Save JSON
    if save_json:
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

    print('Done. (%.3fs)' % (time.time() - t0))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfgs/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov4.weights', help='weights path')
    parser.add_argument('--source', type=str, default='/home/adam/PycharmProjects/PyTorch_YOLOv4_Darknet/data/testdev2017.txt', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--height', type=int, default=512, help='inference height (pixels)')
    parser.add_argument('--width', type=int, default=512, help='inference width (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-img', action='store_true', help="Save results")
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-json', action="store_true", help="save results to json")
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
