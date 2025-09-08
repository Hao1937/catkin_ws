# filepath: /home/clb/catkin_ws/src/colordetector/scripts/img3.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
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
weights_path = '/home/clb/catkin_ws/yolov5/last.pt' 
device = select_device('') # 自动选择CPU或GPU
model = None
names = []
try:
    # 加载模型
    model = attempt_load(weights_path, map_location=device)
    # 获取类别名称
    names = model.module.names if hasattr(model, 'module') else model.names
except Exception as e:
    print(f"加载模型失败: {e}")
    if not os.path.exists(weights_path):
         print(f"提示: 请确保权重文件 '{weights_path}' 存在。")
    exit()

target_classes = ['xuebi', 'kele', 'fenda', 'mozhao']  # 根据您的训练类别调整：雪碧、可乐、芬达、魔爪的拼音标签

# 初始化发布者（移除 pub_image）
pub_json = rospy.Publisher('/detected_objects', String, queue_size=10)

def image_callback(msg):
    """
    处理ROS彩色图像消息的回调函数。
    进行YOLOv5检测，并发布结果。
    """
    try:
        # 使用 numpy 直接转换（假设 BGR8 编码）
        height, width = msg.height, msg.width
        channels = 3  # BGR8
        original_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, channels))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)  # 如果需要调整顺序
    except Exception as e:
        print(f"转换图像失败: {e}")
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
                    
                    # 准备标签文本（无深度信息）
                    label = f'{label_name} {conf:.2f}'
                    
                    # 绘制边界框和标签（可选，移除 cv2.imshow()）
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 添加到结果列表（position 设为 N/A）
                    detected_objects.append({
                        'label': label_name,
                        'bbox': [x1, y1, x2, y2],
                        'position': 'N/A'
                    })

    # --- 步骤7: 发布结果 ---
    # 只发布JSON数据
    pub_json.publish(String(data=json.dumps(detected_objects)))

    # 移除 cv2.imshow() 以避免 GTK 冲突
    # cv2.imshow("YOLO Detection", output_img)
    # cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('yolo_ros_node')
    
    # 订阅图像话题
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback, queue_size=1)
    
    print("YOLO ROS 节点已启动，正在监听图像话题...")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down YOLO ROS node.")
    # 移除 cv2.destroyAllWindows()