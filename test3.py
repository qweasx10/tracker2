from lxml import etree
import cv2
import numpy as np

tree = etree.parse('C007201_002.xml')
root = tree.getroot()
d = root.find('.//Loitering').findall('Point')
e = []
m = []
cnt =0

for p in d:
    v = p.text.split(',')
    e.append(v)
    m.append([])
    for i in range(len(v)):

        m[cnt].append(int(v[i]))
    cnt+=1



print(e)

print(m)





cap = cv2.VideoCapture('../Yolov5_DeepSort_Pytorch/C007201_002_Trim.mp4')
while cap.isOpened():


    ret, frame = cap.read()
    # 프레임이 올바르게 읽히면 ret은 Tru
    if not ret:
        print("프레임을 수신할 수 없습니다(스트림 끝?). 종료 중 ...")
        break
    for i in range(len(e)):
        if i <(len(e)-1):
            cv2.line(frame, (int(e[i][0]), int(e[i][1])), (int(e[i+1][0]), int(e[i+1][1])), (0, 255, 255), 2)
        else:
            cv2.line(frame, (int(e[i][0]), int(e[i][1])), (int(e[0][0]), int(e[0][1])), (0, 255, 255), 2)


    #cv2.line(frame, (int(e[0][0]),int(e[0][1])), (int(e[1][0]),int(e[1][1])), (0, 255, 255), 2)
    #cv2.line(frame, (int(e[1][0]),int(e[1][1])), (int(e[2][0]),int(e[2][1])), (0, 255, 255), 2)
    #cv2.line(frame, (int(e[2][0]),int(e[2][1])), (int(e[3][0]),int(e[3][1])), (0, 255, 255), 2)
    #cv2.line(frame, (int(e[3][0]),int(e[3][1])), (int(e[4][0]),int(e[4][1])), (0, 255, 255), 2)
    #cv2.line(frame, (int(e[4][0]),int(e[4][1])), (int(e[0][0]),int(e[0][1])), (0, 255, 255), 2)
    cv2.imshow('image', frame)

    if cv2.waitKey(42) == ord('q'):
        break
# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()




