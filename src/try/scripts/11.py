#!/usr/bin/env python
# coding:utf-8

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')
target_classes = ['bottle', 'cup']
bridge = CvBridge()

def image_callback(msg):
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    results = model(img)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label_name = model.names[cls]
            if label_name in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    mean_color = roi.mean(axis=(0, 1))
                    color_str = f'RGB({int(mean_color[2])},{int(mean_color[1])},{int(mean_color[0])})'
                else:
                    color_str = 'RGB(0,0,0)'
                label = f'{label_name} {conf:.2f} {color_str}'
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = center_x - w // 2
                text_y = center_y + h // 2
                cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("YOLO Detection", img)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('yolo_camera_node')
    rospy.Subscriber('/camera', Image, image_callback)
    print('YOLO识别节点已启动，正在实时展示结果...')
    rospy.spin()
    cv2.destroyAllWindows()