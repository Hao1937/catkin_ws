#!/usr/bin/env python
# coding: utf-8

# 导入必要的库
import rospy  # ROS Python接口库
import actionlib  # ROS动作库，用于与move_base交互
from sensor_msgs.msg import Image, CameraInfo  # ROS消息类型，用于图像和相机信息
from cv_bridge import CvBridge  # OpenCV与ROS图像消息转换工具
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  # move_base消息类型
import numpy as np  # 数值计算库
import tf  # ROS坐标变换库
from std_msgs.msg import String  # 用于接收检测结果
import json  # 用于解析检测结果

class ColorBottleDetector(object):
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('color_bottle_detector')

        # 初始化工具类
        self.bridge = CvBridge()  # 用于ROS图像消息与OpenCV图像的转换
        self.camera_info = None  # 存储相机内参
        self.tf_listener = tf.TransformListener()  # 用于坐标变换

        # 订阅深度图像和相机内参话题
        self.sub_depth = rospy.Subscriber('/camera/depth/rawimage', Image, self.cb_depth, queue_size=1)
        self.sub_camera_info = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.cb_camera_info, queue_size=1)

        # 订阅目标检测结果
        self.sub_detected_objects = rospy.Subscriber('/detected_objects', String, self.cb_detected_objects, queue_size=1)

        # 定义颜色与目标点的映射关系
        self.color_to_target = {
            'red': (2.0, 0.0),    # 红色瓶子的目标点
            'green': (0.0, 2.0),  # 绿色瓶子的目标点
            'blue': (-2.0, 0.0)   # 蓝色瓶子的目标点
        }

        rospy.loginfo('color_bottle_detector node started')

    def cb_camera_info(self, msg):
        """获取相机内参"""
        if self.camera_info is None:
            self.camera_info = msg

    def cb_depth(self, msg):
        """获取深度图像"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')

    def move_to_goal(self, x, y):
        """使用 move_base 导航到目标点"""
        # 创建move_base客户端
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()  # 等待move_base服务启动

        # 设置目标点
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"  # 使用地图坐标系
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0

        # 发送目标点
        rospy.loginfo(f"导航到目标点: x={x}, y={y}")
        client.send_goal(goal)
        client.wait_for_result()  # 等待导航完成

        # 检查导航结果
        if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("成功到达目标点！")
            return True
        else:
            rospy.logwarn("导航失败！")
            return False

    def pixel_to_camera_coords(self, u, v, depth):
        """将像素坐标转换为相机坐标
        参数：
            u, v: 像素坐标
            depth: 深度值
        返回：
            相机坐标 (x, y, z)
        """
        # 从相机内参中获取焦距和光心
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        # 计算相机坐标
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.array([x, y, z])

    def camera_to_map_coords(self, camera_coords):
        """将相机坐标转换为地图坐标
        参数：
            camera_coords: 相机坐标 (x, y, z)
        返回：
            地图坐标 (x, y, z)
        """
        try:
            # 等待相机到地图的坐标变换
            self.tf_listener.waitForTransform("map", "camera_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "camera_link", rospy.Time(0))

            # 计算变换矩阵
            transform_matrix = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))

            # 将相机坐标转换为齐次坐标
            camera_coords_h = np.append(camera_coords, 1)

            # 应用变换矩阵
            map_coords_h = np.dot(transform_matrix, camera_coords_h)
            return map_coords_h[:3]  # 返回地图坐标
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("无法获取相机到地图的变换")
            return None

    def cb_detected_objects(self, msg):
        """处理目标检测结果"""
        if self.depth_image is None or self.camera_info is None:
            rospy.logwarn("深度图像或相机内参未准备好")
            return

        try:
            detected_objects = json.loads(msg.data)  # 解析检测结果
        except json.JSONDecodeError:
            rospy.logwarn("无法解析检测结果")
            return

        for obj in detected_objects:
            label_name = obj['label']
            x1, y1, x2, y2 = obj['bbox']
            color = obj['color']

            # 获取目标中心点（像素坐标）
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 获取深度值
            depth = self.depth_image[center_y, center_x]
            if np.isnan(depth) or depth <= 0:
                rospy.logwarn("无效的深度值")
                continue

            # 转换为相机坐标
            camera_coords = self.pixel_to_camera_coords(center_x, center_y, depth)
            if camera_coords is None:
                continue
            
            rospy.loginfo(f"目标 '{label_name}' 相对于相机的位置: X={camera_coords[0]:.2f}m, Y={camera_coords[1]:.2f}m, Z={camera_coords[2]:.2f}m")

            # 转换为地图坐标
            map_coords = self.camera_to_map_coords(camera_coords)
            if map_coords is not None:
                map_x, map_y, _ = map_coords
                rospy.loginfo(f"目标 {label_name} 的地图坐标: ({map_x}, {map_y})")

                # 导航到目标点
                rospy.loginfo(f"导航到瓶子位置: ({map_x}, {map_y})")
                if self.move_to_goal(map_x, map_y):
                    destination = self.color_to_target.get(color)
                    if destination:
                        rospy.loginfo(f"导航瓶子到目标点: {destination}")
                        self.move_to_goal(destination[0], destination[1])
                    else:
                        rospy.logwarn(f"未定义颜色 {color} 的目标点")

    def spin(self):
        """保持节点运行"""
        rospy.spin()

if __name__ == '__main__':
    # 创建检测器实例并运行
    detector = ColorBottleDetector()
    detector.spin()