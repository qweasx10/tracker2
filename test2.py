# event_test.py : opencv가 감지할 수 있는 마우스 상태를 dir(cv2) 내부의 'EVENT'를 포함시키는 단어를 출력하면 알 수 있음.
# 계속해서 마우스 상태를 감지하며
# event가 마우스를 누른 상태이면 왼쪽위 좌표 설정, 이동한 상태이면 오른쪽 아래 이동하는동안 계속 설정, 마우스 떼면 마우스 클릭한 상태 변경

import cv2
import numpy as np

# opencv가 감지할 수 있는 mouse event 확인하기
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)




# Mouse Callback함수 : 파라미터는 고정됨.
def draw_rectangle(event, x, y, flags, param):
    global x1, y1, click,x2, y2  # 전역변수 사용

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        print("사각형의 왼쪽위 설정 : (" + str(x1) + ", " + str(y1) + ")")



    elif event == cv2.EVENT_LBUTTONUP:
        click = False;  # 마우스를 때면 상태 변경
        x2, y2 = x,y
        print("사각형의 왼쪽위 설정 : (" + str(x2) + ", " + str(y2) + ")")

    # cv2.circle(img,(x,y),5,(0,255,0),-1)


# 캔버스, MouseCallback 함수 설정
#img = np.zeros((500, 500, 3), np.uint8)
#img = cv2.imread(img_file = "../Yolov5_DeepSort_Pytorch/C007201_002_Trim.mp4")
cap = cv2.VideoCapture('../Yolov5_DeepSort_Pytorch/C007201_002_Trim.mp4')
  # 마우스 이벤트 후 callback 수행하는 함수 지정

# main문 : 키보드로 esc를 받을때까지 화면을 계속 보여준다.
#while True:
#    cv2.imshow('image', cap)  # 화면을 보여준다.

 #   k = cv2.waitKey(1) & 0xFF  # 키보드 입력값을 받고

  #  if k == 27:  # esc를 누르면 종료
   #     break

click = False  # Mouse 클릭된 상태 (false = 클릭 x / true = 클릭 o) : 마우스 눌렀을때 true로, 뗏을때 false로
x1, y1 = -1, -1
x2, y2 = -1, -1
while cap.isOpened():

    cv2.setMouseCallback('image', draw_rectangle)# cap 정상동작 확인
    ret, frame = cap.read()
    # 프레임이 올바르게 읽히면 ret은 True
    if not ret:
        print("프레임을 수신할 수 없습니다(스트림 끝?). 종료 중 ...")
        break
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imshow('image', frame)

    if cv2.waitKey(42) == ord('q'):
        break
# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()