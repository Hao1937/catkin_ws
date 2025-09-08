#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import rospy
# from sensor_msgs.msg import Image, CameraInfo
import cv2
import numpy as np
# from std_msgs.msg import String
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
    model = attempt_load(weights_path, map_location=device)
    # 获取类别名称
    names = model.module.names if hasattr(model, 'module') else model.names
except Exception as e:
    print(f"加载模型失败: {e}")
    if not os.path.exists(weights_path):
         print(f"提示: 请确保权重文件 '{weights_path}' 存在。")
    exit()

target_classes = ['cup', 'bottle']  # 只识别杯子和瓶子

# 添加调试数据：打印类别名称
print(f"模型类别名称: {names}")
print(f"目标类别: {target_classes}")

# 全局变量用于存储最新的深度图和相机内参
latest_depth_img = None
camera_info = None

# 初始化发布者（注释掉）
# pub_json = rospy.Publisher('/detected_objects', String, queue_size=10)
# pub_image = rospy.Publisher('/yolo_output/image', Image, queue_size=10)

def get_dominant_color(roi):
    """获取ROI区域的主要颜色（简化版，使用平均颜色）"""
    if roi.size == 0:
        return "unknown", [0, 0, 0]
    # 计算平均颜色
    avg_color = np.mean(roi, axis=(0, 1))
    # 转换为BGR到RGB
    avg_color = avg_color[::-1]  # BGR to RGB
    # 简单分类颜色
    r, g, b = avg_color
    if r > 150 and g < 100 and b < 100:
        color = "red"
    elif r < 100 and g > 150 and b < 100:
        color = "green"
    elif r < 100 and g < 100 and b > 150:
        color = "blue"
    elif r > 150 and g > 150 and b < 100:
        color = "yellow"
    elif r > 150 and g < 100 and b > 150:
        color = "magenta"
    elif r < 100 and g > 150 and b > 150:
        color = "cyan"
    else:
        color = "unknown"
    return color, avg_color

def depth_callback(msg):
    """回调函数，存储最新的深度图像。"""
    global latest_depth_img
    try:
        # 深度图通常是16位单通道 (16UC1) 或 32位浮点型 (32FC1)
        if msg.encoding == '16UC1':
            latest_depth_img = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            # 添加调试数据：打印深度图像信息
            # print(f"接收到深度图像: 编码={msg.encoding}, 形状={latest_depth_img.shape}, 数据类型={latest_depth_img.dtype}")
        elif msg.encoding == '32FC1':
            latest_depth_img = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            # 添加调试数据：打印深度图像信息
            # print(f"接收到深度图像: 编码={msg.encoding}, 形状={latest_depth_img.shape}, 数据类型={latest_depth_img.dtype}")
        else:
            print(f"未处理的深度图像编码: {msg.encoding}")
            latest_depth_img = None
    except Exception as e:
        print(f"转换深度图像时出错: {e}")
        latest_depth_img = None

def camera_info_callback(msg):
    """回调函数，存储相机内参。"""
    global camera_info
    if camera_info is None:
        camera_info = msg
        # 添加调试数据：打印相机内参
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        print(f"相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print("已接收到相机内参。")

def process_image(original_img):
    """
    处理图像，进行YOLOv5检测，识别杯子和瓶子的颜色，并发布结果。
    """
    # 添加调试数据：打印图像信息
    print(f"处理图像: 形状={original_img.shape}")

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

    # 添加调试数据：打印推理结果
    print(f"推理结果: pred.shape = {pred.shape if pred is not None else 'None'}")

    # --- 步骤6: 后处理和绘制 (YOLOv5) ---
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    # 添加调试数据：打印NMS后的结果
    print(f"NMS后检测结果数量: {len(pred) if pred else 0}")
    if pred:
        for i, det in enumerate(pred):
            print(f"  批次 {i}: 检测到 {len(det)} 个物体")

    detected_objects = []
    output_img = original_img.copy()

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_img.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label_name = names[int(cls)]
                
                # 添加调试数据：打印所有检测到的类别
                print(f"检测到类别: {label_name}, 置信度: {conf:.2f}")
                
                # 只处理目标类别
                if label_name in target_classes:
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # 获取ROI并识别颜色
                    roi = original_img[y1:y2, x1:x2]
                    color, avg_color = get_dominant_color(roi)
                    
                    # 准备标签文本，包含颜色
                    label = f'{label_name} {color} {conf:.2f}'
                    
                    # 绘制边界框和标签
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 添加调试数据
                    print(f"识别成功: 类别={label_name}, 颜色={color}, 置信度={conf:.2f}, 边界框=[{x1},{y1},{x2},{y2}]")
                    
                    # 添加到结果列表，包含颜色信息
                    detected_objects.append({
                        'label': label_name,
                        'color': color,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf)
                    })

    # --- 步骤7: 发布结果 ---
    # 添加调试数据
    print(f"检测到 {len(detected_objects)} 个目标对象")
    # 发布JSON数据（注释掉）
    # pub_json.publish(String(data=json.dumps(detected_objects)))
    
    # 发布带边界框的图像（注释掉）
    # try:
    #     img_msg = Image()
    #     img_msg.header = msg.header  # 使用原始消息的header
    #     img_msg.height = output_img.shape[0]
    #     img_msg.width = output_img.shape[1]
    #     img_msg.encoding = "bgr8"
    #     img_msg.is_bigendian = False
    #     img_msg.step = output_img.shape[1] * 3
    #     img_msg.data = output_img.tobytes()
    #     pub_image.publish(img_msg)
    # except Exception as e:
    #     print(f"发布图像时出错: {e}")

    # --- 步骤8: 实时显示结果 ---
    #cv2.imshow("YOLO Detection", output_img)
    #cv2.waitKey(0)  # 按任意键关闭
    
    # 保存结果图像
    result_path = os.path.join(os.path.dirname(__file__), 'result.jpg')
    cv2.imwrite(result_path, output_img)
    print(f"结果图像已保存到: {result_path}")

if __name__ == '__main__':
    # 读取同目录下的1.jpg
    image_path = os.path.join(os.path.dirname(__file__), '1.jpg')
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        exit()
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"无法读取图像: {image_path}")
        exit()
    
    print(f"成功读取图像: {image_path}, 形状={original_img.shape}")
    
    # 处理图像
    process_image(original_img)
