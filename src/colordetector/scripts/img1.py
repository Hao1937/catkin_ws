#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String
import json
import torch
import os
import sys

# --- 步骤1: 将YOLOv5源码目录添加到Python路径中 ---
yolov5_path = '/home/clb/catkin_ws/yolov5'
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# --- 步骤2: 从YOLOv5源码中直接导入模型 ---
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# --- 全局变量和模型加载 ---
# 指定权重文件的绝对路径
weights_path = '/home/clb/catkin_ws/yolov5/yolov5s.pt' 
device = select_device('') # 自动选择CPU或GPU
model = None
names = []
try:
    # 加载模型
    model = attempt_load(weights_path, map_location=device)
    # 获取类别名称
    names = model.module.names if hasattr(model, 'module') else model.names
except Exception as e:
    rospy.logerr(f"加载模型失败: {e}")
    if not os.path.exists(weights_path):
         rospy.logerr(f"提示: 请确保权重文件 '{weights_path}' 存在。")
    exit()

target_classes = ['bottle', 'cup']
bridge = CvBridge()

# 初始化发布者
pub_json = rospy.Publisher('/detected_objects', String, queue_size=10)
pub_image = rospy.Publisher('/yolo_output/image', Image, queue_size=10)

def image_callback(msg):
    """
    处理ROS图像消息的回调函数。
    进行YOLOv5检测，并发布JSON结果和带边界框的图像。
    """
    try:
        original_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr(f"无法将ROS图像消息转换为OpenCV图像: {e}")
        return

    # --- 步骤4: 准备图片 (YOLOv5) ---
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # --- 步骤5: 运行推理 (YOLOv5) ---
    pred = model(img, augment=False)[0]

    # --- 步骤6: 后处理和绘制 (YOLOv5) ---
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    detected_objects = []
    output_img = original_img.copy()

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label_name = names[int(cls)]
                
                if label_name in target_classes:
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # 计算颜色
                    roi = output_img[y1:y2, x1:x2]
                    if roi.size > 0:
                        mean_color = roi.mean(axis=(0, 1))
                        color_str = f'RGB({int(mean_color[2])},{int(mean_color[1])},{int(mean_color[0])})'
                    else:
                        color_str = 'RGB(0,0,0)'
                    
                    # 准备标签文本
                    label = f'{label_name} {conf:.2f} {color_str}'
                    
                    # 绘制边界框和标签
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_x = center_x - w // 2
                    text_y = center_y + h // 2
                    cv2.putText(output_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 添加到结果列表
                    detected_objects.append({
                        'label': label_name,
                        'bbox': [x1, y1, x2, y2],
                        'color': color_str
                    })

    # --- 步骤7: 发布结果 ---
    # 发布JSON数据
    pub_json.publish(String(data=json.dumps(detected_objects)))
    
    # 发布带边界框的图像
    try:
        img_msg = bridge.cv2_to_imgmsg(output_img, "bgr8")
        pub_image.publish(img_msg)
    except Exception as e:
        rospy.logerr(f"发布图像时出错: {e}")

if __name__ == '__main__':
    rospy.init_node('yolo_camera_node')
    # 订阅摄像头话题
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    rospy.loginfo("YOLOv5识别节点已启动，正在监听 /camera/color/image_raw ...")
    rospy.loginfo("要查看结果图像，请运行: rqt_image_view /yolo_output/image")
    rospy.spin()
