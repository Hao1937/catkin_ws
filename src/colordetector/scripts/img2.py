#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image, CameraInfo
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
    rospy.logerr(f"加载模型失败: {e}")
    if not os.path.exists(weights_path):
         rospy.logerr(f"提示: 请确保权重文件 '{weights_path}' 存在。")
    exit()

target_classes = ['xuebi', 'kele', 'fenda', 'mozhao']  # 根据您的训练类别调整：雪碧、可乐、芬达、魔爪的拼音标签
bridge = CvBridge()

# 添加调试数据：打印类别名称
print(f"模型类别名称: {names}")
print(f"目标类别: {target_classes}")

# 全局变量用于存储最新的深度图和相机内参
latest_depth_img = None
camera_info = None

# 初始化发布者
pub_json = rospy.Publisher('/detected_objects', String, queue_size=10)
pub_image = rospy.Publisher('/yolo_output/image', Image, queue_size=10)

def depth_callback(msg):
    """回调函数，存储最新的深度图像。"""
    global latest_depth_img
    try:
        # 深度图通常是16位单通道 (16UC1) 或 32位浮点型 (32FC1)
        if msg.encoding == '16UC1':
            latest_depth_img = bridge.imgmsg_to_cv2(msg, "16UC1")
            # 添加调试数据：打印深度图像信息
            print(f"接收到深度图像: 编码={msg.encoding}, 形状={latest_depth_img.shape}, 数据类型={latest_depth_img.dtype}")
        elif msg.encoding == '32FC1':
            latest_depth_img = bridge.imgmsg_to_cv2(msg, "32FC1")
            # 添加调试数据：打印深度图像信息
            print(f"接收到深度图像: 编码={msg.encoding}, 形状={latest_depth_img.shape}, 数据类型={latest_depth_img.dtype}")
        else:
            rospy.logwarn_throttle(5, f"未处理的深度图像编码: {msg.encoding}")
            latest_depth_img = None
    except Exception as e:
        rospy.logerr(f"转换深度图像时出错: {e}")
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
        rospy.loginfo("已接收到相机内参。")

def image_callback(msg):
    """
    处理ROS彩色图像消息的回调函数。
    进行YOLOv5检测，计算相对位置，并发布结果。
    """
    # 添加调试数据：打印图像信息
    print(f"接收到图像: 宽度={msg.width}, 高度={msg.height}, 编码={msg.encoding}")
    
    global latest_depth_img, camera_info
    if latest_depth_img is None:
        rospy.logwarn_throttle(5, "正在等待深度图像...")
        return
    if camera_info is None:
        rospy.logwarn_throttle(5, "正在等待相机内参...")
        return

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
                
                if label_name in target_classes:
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # --- 获取深度和计算3D坐标 ---
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # 确保坐标在深度图范围内
                    if 0 <= center_y < latest_depth_img.shape[0] and 0 <= center_x < latest_depth_img.shape[1]:
                        depth = latest_depth_img[center_y, center_x]
                        
                        # 处理深度值（16UC1单位为mm，32FC1单位为m）
                        if latest_depth_img.dtype == np.uint16:
                            depth_m = depth / 1000.0
                        else:
                            depth_m = depth

                        if depth_m > 0:
                            # 从相机内参获取焦距和光心
                            fx = camera_info.K[0]
                            fy = camera_info.K[4]
                            cx = camera_info.K[2]
                            cy = camera_info.K[5]

                            # 计算相机坐标
                            cam_x = (center_x - cx) * depth_m / fx
                            cam_y = (center_y - cy) * depth_m / fy
                            cam_z = depth_m
                            
                            # 添加调试数据：打印深度和坐标信息
                            print(f"中心点: ({center_x}, {center_y}), 深度值: {depth} mm ({depth_m:.2f} m)")
                            print(f"计算的相机坐标: x={cam_x:.2f}, y={cam_y:.2f}, z={cam_z:.2f}")
                            
                            pos_str = f"Pos:({cam_x:.2f},{cam_y:.2f},{cam_z:.2f})m"
                            print(f"位置字符串: {pos_str}")
                            rospy.loginfo(f"检测到 '{label_name}': {pos_str}")
                        else:
                            pos_str = "Pos: Invalid depth"
                            print(f"无效深度值: {depth} (中心点: {center_x}, {center_y})")
                    else:
                        pos_str = "Pos: Out of bounds"
                        print(f"中心点超出范围: ({center_x}, {center_y}), 深度图形状: {latest_depth_img.shape}")

                    # 准备标签文本
                    label = f'{label_name} {conf:.2f} {pos_str}'
                    
                    # 绘制边界框和标签
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 添加调试数据
                    print(f"识别成功: 类别={label_name}, 置信度={conf:.2f}, 边界框=[{x1},{y1},{x2},{y2}]")
                    
                    # 添加到结果列表
                    detected_objects.append({
                        'label': label_name,
                        'bbox': [x1, y1, x2, y2],
                        'position': pos_str
                    })

    # --- 步骤7: 发布结果 ---
    # 添加调试数据
    print(f"检测到 {len(detected_objects)} 个目标对象")
    # 发布JSON数据
    pub_json.publish(String(data=json.dumps(detected_objects)))
    
    # 发布带边界框的图像
    try:
        img_msg = bridge.cv2_to_imgmsg(output_img, "bgr8")
        pub_image.publish(img_msg)
    except Exception as e:
        rospy.logerr(f"发布图像时出错: {e}")

    # --- 步骤8: 实时显示结果 ---
    cv2.imshow("YOLO Detection", output_img)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('yolo_ros_node')
    
    # 订阅图像话题
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback, queue_size=1)
    # 订阅深度图像话题
    rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback, queue_size=1)
    # 订阅相机内参话题
    rospy.Subscriber('/camera/depth/camera_info', CameraInfo, camera_info_callback, queue_size=1)
    
    rospy.loginfo("YOLO ROS 节点已启动，正在监听图像话题...")
    rospy.spin()
