import sys
import math
sys.path.insert(0, './yolov5')
from xml.etree.ElementTree import Element, SubElement, ElementTree
from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import datetime
from lxml import etree



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def xml(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    d = root.find('.//Intrusion').findall('Point')
    return d

def is_inside(polygon, point):

    def cross(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        if y1-y2 == 0:
            if y1 == point[1]:
                if min(x1, x2) <= point[0] <= max(x1, x2):
                    return 1, True
            return 0, False

        if x1 - x2 == 0:
            if min(y1, y2) <= point[1] <= max(y1, y2):
                if point[0] <= max(x1, x2):
                    return 1, point[0] == max(x1, x2)
            return 0, False

        a = (y1 - y2) / (x1 - x2)
        b = y1 - x1 * a
        x = (point[1] - b) / a
        if point[0] <= x:
            if min(y1, y2) <= point[1] <= max(y1, y2):
                return 1, point[0] == x or point[1] in (y1,y2)
        return 0, False

    cross_points = 0
    for x in range(len(polygon)):
        num, on_line = cross(polygon[x], polygon[x-1])
        if on_line:
            return True
        cross_points += num
    return cross_points % 2

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def makexml(filename,alt,frame_idx):

    root = Element('KisaLibraryIndex')
    lib = SubElement(root,'Library')
    SubElement(lib,'Senario').text = 'Intrusion'
    SubElement(lib,'Dataset').text = 'KISA201'
    SubElement(lib,'Libversion').text = '1.0'
    clip = SubElement(lib,'Clip')
    head = SubElement(clip,'Header')
    SubElement(head,'Filename').text = filename+'.mp4'
    SubElement(head,'Duration').text =(lambda x : '0'+x if len(x) < 8 else x)(str(datetime.timedelta(seconds = frame_idx)))
    SubElement(head, 'AlarmEvents').text ='1'
    al = SubElement(clip,'Alarms')
    alm = SubElement(al,'Alarm')
    SubElement(alm,'StartTime').text = (lambda x : '0'+x if len(x) < 8 else x)(str(datetime.timedelta(seconds = alt)))
    SubElement(alm,'AlarmDescription').text = 'Intrusion'
    SubElement(alm,'AlarmDuration').text = '00:00:10'


    tree = ElementTree(root)
    tree.write('./' + filename + '.xml')
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls_names, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '%d %s' % (id, cls_names[i])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        y_h = int(y1+(y2-y1)/2)
        x_h = int(x1+(x2-x1)/2)
        cv2.line(img, (x_h, y_h), (x_h, y_h), (0, 0, 255), 5)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    #return img


def myInterSibal(person_corX, person_corY):
    # ????????? if ????????? ?????? ?????? ????????? ???, ?????? ?????? ?????? ???????????? ??????????????? ?????????.
    # myline?????? ?????????????????????735
    #line = [(735, 520), (1093, 287)]
    mylineX1 = 735
    mylineY1 = 520
    mylineX2 = 1093
    mylineY2 = 287
    #?????? ????????? Lgiul

    Lgiul = (mylineY2-mylineY1)/(mylineX2-mylineX1)

    #?????? ?????? ????????? spot_Lgiul
    spot_Lgiul1 = (person_corY-mylineY1)/(person_corX-mylineX1)
    spot_Lgiul2 = (mylineY2-person_corY)/(mylineX2-person_corX)
    c = math.degrees(2 * math.pi)

    if Lgiul < 0:
        if spot_Lgiul2 > Lgiul and person_corY>mylineY2 and person_corY<mylineY1:
            return True
        else: return False

    elif Lgiul > 0:
        if spot_Lgiul2 < Lgiul:
            return True
        else:
            return False



def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    cnt = 1
    # initialize deepsort
    counter = 0
    line = [(735, 520), (1093, 287)]
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)



    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            e = []
            pol = []
            ct = 0
            for o in xml('C104301_001.xml'):
                v = o.text.split(',')
                e.append(v)
                pol.append([])
                for l in range(len(v)):
                    pol[ct].append(int(v[l]))
                ct += 1
            for i in range(len(e)):
                if i < (len(e) - 1):
                    cv2.line(im0, (int(e[i][0]), int(e[i][1])), (int(e[i + 1][0]), int(e[i + 1][1])), (0, 255, 255),
                             2)
                else:
                    cv2.line(im0, (int(e[i][0]), int(e[i][1])), (int(e[0][0]), int(e[0][1])), (0, 255, 255), 2)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string


                xywh_bboxs = []
                confs = []
                classes= []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)

                    confs.append([conf.item()])
                    classes.append([cls.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)


                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, classes, im0)
                x_h = int(xywh_obj[0]+xywh_obj[2]/2) #????????? ??????
                y_h = int(xywh_obj[1]+xywh_obj[3]/2)
                x_o = int(xywh_obj[0]-xywh_obj[2]/2) #?????? ??????
                y_o = int(xywh_obj[1]-xywh_obj[3]/2)
                x_r = int(xywh_obj[1]-xywh_obj[3]/2) #????????? ??????
                y_r = int(xywh_obj[0]+xywh_obj[2]/2)
                x_l = int(xywh_obj[0]-xywh_obj[2]/2)
                y_l = int(xywh_obj[1]+xywh_obj[3]/2)
                x_k = int(xywh_obj[0])
                y_k = int(xywh_obj[1])
                p1 = (int(xywh_obj[0]),int(xywh_obj[1]))

                # ?????? ?????????
                point = [x_k, y_k]
                point1 = [x_h, y_h]
                point2 = [x_o, y_o]
                point3 = [x_r, y_r]
                point4 = [x_l, y_l]
                if is_inside(pol, point):
                        cv2.putText(im0,'**Invader**', (50, 75), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 3)


                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    classes = outputs[:, 5]
                    draw_boxes(im0, bbox_xyxy,[names[i] for i in classes], identities)
                    # to MOT format
                    #cv2.line(im0, (x_h, y_h), (x_h, y_h), (0, 0, 255), 5)
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # Write MOT compliant results to file


                    if save_txt:


                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            cnt += 1
                            if cnt == 2:

                                a = frame_idx
                                print(a)
                                alt = int(a / 30)
                            td = int(frame_idx/30)
                            #makexml(txt_file_name, alt, td)
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                            bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/gg.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', default='0', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
