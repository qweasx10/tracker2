# Yolov5 + Deep Sort with PyTorch




#실행방법
1. pycharm을 설치
2. anaconda prompt 창켜서 가상환경 만든다 conda create -n 원하는이름 python=3.8
3. conda activate 만든가상환경이름
4. 파이참으로 넘어와서 만든 가상환경 open해주고 settings 들어가서 python interpreter누르고 우측상단에 설정모양 누르고 add누르고 conda enviroment 누르고 existing enviroment 누르고 가상환경안의 python 선택 후 저장
5. 이제 터미널 창에 git clone https://github.com/qweasx10/tracker2.git
6. 그후 cd Yolov5_DeepSort_Pytorch
7. 그후 pip install -r requirements.txt
8. 그리고 cuda 설치 conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
9. 실행파일이 2개 있는데 tracker.py는 기존 Invade.py는 침입임 
10. 실행해보자 python 원하는파일.py --device 0 --source 0 --show-vid --save-txt
11. --device 0(gpu 사용한다는거) --source 0(캠) or 파일이름.mp4(이러면 저장된 비디오 재생) --show-vid(모니터링) --save-txt(결과 텍스트파일로 저장)
12. 그렇군
<div align="center">
<p>
<img src="MOT16_eval/KakaoTalk_20220124_180117100.png" width="400"/> <img src="MOT16_eval/track_all.gif" width="400"/> 
</p>
<br>
<div>
<a href="https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/actions"><img src="https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
<br>  
<a href="https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 
</div>

</div>


## Introduction

This repository contains a two-stage-tracker. The detections generated by [YOLOv5](https://github.com/ultralytics/yolov5), a family of object detection architectures and models pretrained on the COCO dataset, are passed to a [Deep Sort algorithm](https://github.com/ZQPei/deep_sort_pytorch) which tracks the objects. It can track any object that your Yolov5 model was trained to detect.


## Tutorials

* [Yolov5 training on Custom Data (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
* [Deep Sort deep descriptor training (link to external repository)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)&nbsp;
* [Yolov5 deep_sort pytorch evaluation](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation)&nbsp;



## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`


## Tracking sources

Tracking can be run on most video formats

```bash
python3 track.py --source ... --show-vid  # show live inference results as well
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`


## Select a Yolov5 family model

There is a clear trade-off between model inference speed and accuracy. In order to make it possible to fulfill your inference speed/accuracy needs
you can select a Yolov5 family model for automatic download

```bash
python3 track.py --source 0 --yolo_weights yolov5s.pt --img 640  # smallest yolov5 family model
```

```bash
python3 track.py --source 0 --yolo_weights yolov5x6.pt --img 1280  # largest yolov5 family model
```


## Filter tracked classes

By default the tracker tracks all MS COCO classes.

If you only want to track persons I recommend you to get [these weights](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) for increased performance

```bash
python3 track.py --source 0 --yolo_weights yolov5/weights/crowdhuman_yolov5m.pt --classes 0  # tracks persons, only
```

If you want to track a subset of the MS COCO classes, add their corresponding index after the classes flag

```bash
python3 track.py --source 0 --yolo_weights yolov5s.pt --classes 16 17  # tracks cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov5 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero.


## MOT compliant results

Can be saved to `inference/output` by 

```bash
python3 track.py --source ... --save-txt
```


## Cite

If you find this project useful in your research, please consider cite:

```latex
@misc{yolov5deepsort2020,
    title={Real-time multi-object tracker using YOLOv5 and deep sort},
    author={Mikel Broström},
    howpublished = {\url{https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch}},
    year={2020}
}
```


## Other information

For more detailed information about the algorithms and their corresponding lisences used in this project access their official github implementations.

