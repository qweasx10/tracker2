# 현 + 범
# 사용 명령어
# python Usethis.py --source Intrusion.mp4 --img 640



import sys


sys.path.insert(0, './yolov5')

from xml.etree.ElementTree import Element, SubElement, ElementTree
from lxml import etree

from numpy import random



from yolov5.utils.plots import plot_one_box
from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow, xyxy2xywh
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
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
유기 = './/DetectArea'
침입 = './/Intrusion'
배회 = './/Loitering'

#######################################여기서 상황 쓰셈
circumstance = 배회

#######################################


# 쓰고싶은 xml 쓰기

# xmlname = 'abandon.xml'
# xmlname = 'Intrusion.xml'
xmlname = 'C002101_001.xml'

#유기판단 함수
def abandonment(point, point_2):
    # obj: point_2 좌표
    # 사람: point 좌표
    # 거리 계산(전체 크기에서 x 비율 잡고, 그 비율보다 작으면 잡아내는것으로 만들어야 함)

    if (abs(point[0] - point_2[0]) <= 10)&(abs(point[1] - point_2[1]) <= 10):
        print("##########################close#################################")
    else:
        pass


def is_inside(polygon, point):
    def cross(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        if y1 - y2 == 0:
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
                return 1, point[0] == x or point[1] in (y1, y2)
        return 0, False

    cross_points = 0
    for x in range(len(polygon)):
        num, on_line = cross(polygon[x], polygon[x - 1])
        if on_line:
            return True
        cross_points += num
    return cross_points % 2


def xyxy_to_xywh(*xyxy):
    '''Calculates the relative bounding box from absolute pixel values. '''
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

def xml(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    d = root.find(circumstance).findall('Point')
    return d

#현민아 부탁할게. makexml 까지 하는구나
def makexml(filename, alt, frame_idx):
    root = Element('KisaLibraryIndex')
    lib = SubElement(root, 'Library')
    SubElement(lib, 'Senario').text = 'Intrusion'
    SubElement(lib, 'Dataset').text = 'KISA201'
    SubElement(lib, 'Libversion').text = '1.0'
    clip = SubElement(lib, 'Clip')
    head = SubElement(clip, 'Header')
    SubElement(head, 'Filename').text = filename + '.mp4'
    SubElement(head, 'Duration').text = (lambda x: '0' + x if len(x) < 8 else x)(
        str(datetime.timedelta(seconds=frame_idx)))
    SubElement(head, 'AlarmEvents').text = '1'
    al = SubElement(clip, 'Alarms')
    alm = SubElement(al, 'Alarm')
    SubElement(alm, 'StartTime').text = (lambda x: '0' + x if len(x) < 8 else x)(str(datetime.timedelta(seconds=alt)))
    SubElement(alm, 'AlarmDescription').text = 'Intrusion'
    SubElement(alm, 'AlarmDuration').text = '00:00:10'

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
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)

        # My#
        y_h = int(y1 + (y2 - y1) / 2)
        x_h = int(x1 + (x2 - x1) / 2)

        # 추출된 원하는 객체만 가운데에 점찍기.
        if cls_names[i] == 'person':
            cv2.line(img, (x_h, y_h), (x_h, y_h), (0, 0, 255), 5)
        # My#

        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return x_h, y_h




def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')



    # initialize deepsort
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
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports images displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant images size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    # My #
    # 여기에 써야 한번만 읽음

    # 원하는 클래스만 계산

    idx_person = names.index("person")

    # My #
    ctt = 0
    # for문 돌아가며 인식 계속 돌아가는 for문으로 객체를 인식하고, 바운딩 박스를 for문마다 그려줌
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

        # Process detections (Apply Classifier)
        for i, det in enumerate(pred):  # detections per images
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)
            e = []
            pol = []
            ct = 0

            # xml 파일 지정
            for o in xml(xmlname):
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

            # 공통영역. 식별 시작
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
                classes = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]

                    #  cls.item() == 0.0 << 사람의 경우가 됨. car의 경우 2.0 # 인덱스의 번호를 따름 그렇지만 사용하면 오류
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    classes.append([cls.item()])
                    # classes == [[0.0]] << 사람의 경우가 됌 car의 경우 [[2.0]] # 응 안써

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)
                # pass detections to deepsort -># draw boxes for visualization 넘어감(메인 코드)
                # 추출된 객체의 모든 정보를 딥솔트해서  >> outputs로 넘겨줌
                outputs = deepsort.update(xywhs, confss, classes, im0)

                #ㅇㅇㅇㅇㅇ 지금 pending
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    classes = outputs[:, -1]

                    obj = [names[i] for i in classes]

                    ## 원하는 obj의 가운데 좌표 얻어냄
                    if obj != 'person':
                        obj_x, obj_y = draw_boxes(im0, bbox_xyxy, [names[i] for i in classes], identities)

                    else:
                        draw_boxes(im0, bbox_xyxy, [names[i] for i in classes], identities)





                # My 사람만 인지하여, 파라미터 뽑아내는 부분 #
                # Deep SORT: person class only
                idxs_ppl = (det[:, -1] == idx_person).nonzero(as_tuple=False).squeeze(
                    dim=1)  # 1. List of indices with 'person' class detections
                dets_ppl = det[idxs_ppl, :-1]  # 2. Torch.tensor with 'person' detections
                print('\n {} people were detected!'.format(len(idxs_ppl)))

                # Deep SORT: convert data into a proper format
                Per_xywhs = xyxy2xywh(dets_ppl[:, :-1]).to("cpu")
                Per_confs = dets_ppl[:, 4].to("cpu")

                # My 식별된 person만 인지#
                if len(dets_ppl) != 0:
                    trackers = deepsort.update(Per_xywhs, Per_confs, classes, im0)
                    #나중에 써먹을 일 없으면 지우기, 써져있는 박스 위에 하나 더 쓰는거
                    for d in trackers:
                        plot_one_box(d[:-1], im0, label='ID' + str(int(d[-1])), color=colors[1], line_thickness=1)
                    if len(trackers) > 0:
                        Person_bbox_xyxy = trackers[:, :4]
                        Per_identities = trackers[:, 4]
                        Per_classes = trackers[:, -1]

                        x_h, y_h = draw_boxes(im0, Person_bbox_xyxy, [names[i] for i in Per_classes], Per_identities)
                        # x_h, y_h 가 사람 클래스만의 점이 됨.
                        # My #

                        # !! point 는 사람 클래스의 가운데 점
                        point = [x_h, y_h]

                        # !! point_2 는 사람 외의 클래스의 가운데 점
                        point_2 = [obj_x, obj_y]

                        ## xml 가져와서 범위 적용. 상황에 맞게 판단 부분.
                        if is_inside(pol, point):
                            ctt+=1
                            if ctt ==1:
                                a = frame_idx
                            if circumstance == 유기:
                                cv2.putText(im0, '*Abandoment Danger Area*', (50, 75), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 3)
                                #'해 줘'
                                abandonment(point, point_2)
                            elif circumstance == 배회 and frame_idx>a+300:
                                cv2.putText(im0, '*Suspious*', (50, 75), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 3)
                            elif circumstance == 침입:
                                cv2.putText(im0, '*Invader*', (50, 75), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 3)




            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (images with detections)
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


# 기본으로 지정'--yolo_weights' '--show-vid' '--device'
# 사용 명령어     python test1.py --source invade.mp4 --img 640
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5m.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', default='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
