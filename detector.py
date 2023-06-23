import cv2
import numpy as np
import os


def get_person_roi(image):
    # YOLO 모델과 관련된 파일 경로
    yolo_weights = 'D:/git/new_stabil/yolo/yolov3.weights'
    yolo_config = 'D:/git/new_stabil/yolo/yolov3.cfg'
    yolo_classes = 'D:/git/new_stabil/yolo/coco.names'

    net = cv2.dnn.readNet(yolo_weights, yolo_config)
    classes = []
    with open(yolo_classes, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1]
                     for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    image = cv2.resize(image, (1920, 1080))

    height, width, channels = image.shape

    blob = cv2.dnn.blobFromImage(
        image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # 객체 검출 수행
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 검출된 객체 필터링
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 사람 클래스의 confidence threshold를 0.5로 설정
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # 비최대 억제 수행
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # ROI 추출
    font = cv2.FONT_HERSHEY_PLAIN
    center_x = width // 2
    center_y = height // 2
    min_dist = float('inf')
    closest_roi = None
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            roi_center_x = x + w // 2
            roi_center_y = y + h // 2
            dist = np.sqrt((roi_center_x - center_x) ** 2 +
                           (roi_center_y - center_y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_roi = [x, y, w, h]

            # label = str(classes[class_ids[i]])
            # color = colors[i]
            # cv2.putText(image, label, (x, y + 30), font, 3, color, 3)
    cv2.rectangle(image, closest_roi, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(1)


def process_dir(dir):

    for filename in os.listdir(dir):
        if filename.endswith(".png"):
            filepath = os.path.join(dir, filename)

            image = cv2.imread(filepath)
            get_person_roi(image)


# 이미지 로드
# image = cv2.imread('356.png')

# 사람 ROI 추출
# get_person_roi(image)

image_dir = 'D:/git/new_stabil/frame'

process_dir(image_dir)

cv2.destroyAllWindows()
