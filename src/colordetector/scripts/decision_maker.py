#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion
import tf
import json
import re
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import numpy as np
#TODO:
# --- 1. 定义关键位置和姿态 ---
# !!! 重要：请根据您的机器人和场地标定这些坐标值 !!!
TARGET_POSITIONS = {
    "left":      (1.0, 0.5, 0.0),  # (x, y, z)
    "right":     (1.0, -0.5, 0.0),
    "front_left":  (1.5, 0.2, 0.0),
    "front_right": (1.5, -0.2, 0.0),
    "starting_point": (0.0, 0.0, 0.0) # 初始观察点
}

# 观察罐子时的标准朝向 (四元数)，请根据场地设置，(0,0,0,1)表示朝向x轴正方向
OBSERVATION_ORIENTATION = Quaternion(0, 0, 0, 1.0) 

# --- 2. 定义颜色到目标区域的映射 ---
COLOR_TO_TARGET_MAP = {
    "red":    "left",
    "green":  "right",
    "blue":   "front_left",
    "yellow": "front_right"
}

# --- 3. 定义标签到目标区域的映射（针对拼音标签） ---
LABEL_TO_TARGET_MAP = {
    "xuebi": "right",    # 雪碧 -> 绿色 -> right
    "kele":  "left",     # 可乐 -> 红色 -> left
    "fenda": "front_right", # 芬达 -> 橙色 -> front_right
    "mozhao": "front_left"  # 魔爪 -> 黑色 -> front_left（或根据需要调整）
}

# --- 全局变量 ---
tf_listener = None
move_base_client = None # 使用actionlib客户端
processed_objects = set() # 存储已处理过的物体
current_observation_pose = np.array(TARGET_POSITIONS["starting_point"]) # 机器人当前的观察位置
TRAVERSAL_STEP = 0.3 # 每次向右移动的步长 (米)

# --- 状态机变量 ---
current_state = 'cruising'  # 初始状态：巡航
STATES = ['cruising', 'pushing', 'returning']

def get_color_name(r, g, b):
    """根据RGB值判断颜色名称"""
    # TODO: 这是一个简化的颜色判断，您需要根据实际情况调整阈值
    # 黑色 (魔爪)
    if r < 50 and g < 50 and b < 50:
        return "black"
    # 红色 (可口可乐)
    if r > 150 and g < 100 and b < 100:
        return "red"
    # 绿色 (雪碧)
    if g > 130 and r < 100 and b < 100:
        return "green"
    # 橙色 (芬达)
    if r > 180 and g > 80 and g < 180 and b < 70:
        return "orange"
    return "unknown"

def parse_position(pos_string):
    """从 'Pos: (x, y, z)' 格式的字符串中解析出坐标值"""
    try:
        match = re.search(r'\(([^)]+)\)', pos_string)
        if match:
            parts = match.group(1).split(',')
            if len(parts) == 3:
                return tuple(map(float, parts))
    except Exception as e:
        rospy.logwarn("解析位置字符串失败: {}, 错误: {}".format(pos_string, e))
    return None

def cruise(event):
    """巡航函数：定期移动观察点"""
    global current_state, current_observation_pose
    if current_state == 'cruising':
        next_observation_pose = current_observation_pose.copy()
        next_observation_pose[1] -= TRAVERSAL_STEP  # 向右移动
        if move_to_goal(next_observation_pose, OBSERVATION_ORIENTATION):
            current_observation_pose = next_observation_pose
            rospy.loginfo("巡航：移动到新观察位置: {}".format(current_observation_pose))
        else:
            rospy.logwarn("巡航：移动到观察点失败！")

def move_to_goal(goal_coords, orientation=Quaternion(0,0,0,1.0)):
    """使用move_base action导航到目标点（包含位置和姿态）"""
    global move_base_client
    if not move_base_client:
        rospy.logerr("Move Base action client 未初始化!")
        return False

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = goal_coords[0]
    goal.target_pose.pose.position.y = goal_coords[1]
    goal.target_pose.pose.orientation = orientation

    rospy.loginfo("发送导航目标: 位置={}, 朝向={}".format(goal_coords, orientation.w))
    move_base_client.send_goal(goal)
    
    finished_within_time = move_base_client.wait_for_result(rospy.Duration(60.0)) 

    if not finished_within_time:
        move_base_client.cancel_goal()
        rospy.logwarn("导航超时！")
        return False
    else:
        state = move_base_client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("成功到达目标点！")
            return True
        else:
            rospy.logwarn("导航失败，状态码: {}".format(state))
            return False

def objects_callback(msg):
    """
    核心回调函数，接收检测结果，转换坐标，并执行完整的导航和操作序列。
    """
    global tf_listener, processed_objects, current_observation_pose, current_state

    if current_state != 'cruising':
        return  # 如果不在巡航状态，忽略检测结果

    try:
        detected_objects = json.loads(msg.data)
    except json.JSONDecodeError:
        rospy.logerr("接收到的数据不是有效的JSON格式")
        return

    if not detected_objects:
        return

    # --- 3. 查找最近的、未处理的物体 ---
    closest_object = None
    min_distance = float('inf')

    for obj in detected_objects:
        obj_id = "{}_{}".format(obj.get('label', 'unknown'), obj.get('position', ''))
        if obj_id in processed_objects:
            continue

        pos_3d_camera = parse_position(obj.get('position', ''))
        if pos_3d_camera:
            # 将相机坐标转换为地图坐标
            try:
                tf_listener.waitForTransform("map", "camera_link", rospy.Time(0), rospy.Duration(1.0))
                (trans, rot) = tf_listener.lookupTransform("map", "camera_link", rospy.Time(0))
                transform_matrix = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))
                camera_coords_h = list(pos_3d_camera) + [1]
                map_coords_h = np.dot(transform_matrix, camera_coords_h)
                map_coords = map_coords_h[:3]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn("坐标变换失败: {}".format(e))
                continue

            # 使用地图坐标系下的距离进行判断
            distance = np.linalg.norm(map_coords - current_observation_pose)
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
                closest_object['id'] = obj_id
                closest_object['map_coords'] = map_coords

    if not closest_object:
        rospy.loginfo("没有发现新的待处理物体。")
        return

    label = closest_object.get('label', 'unknown')
    color_str = closest_object.get('color', 'RGB(0,0,0)')
    r, g, b = list(map(int, re.findall(r'\d+', color_str)))
    color_name = get_color_name(r, g, b)

    # 根据标签或颜色决定目标区域
    target_zone = None
    if label in LABEL_TO_TARGET_MAP:
        target_zone = LABEL_TO_TARGET_MAP[label]
        decision_info = "根据标签 '{}'".format(label)
    elif color_name in COLOR_TO_TARGET_MAP:
        target_zone = COLOR_TO_TARGET_MAP[color_name]
        decision_info = "根据颜色 '{}'".format(color_name)
    else:
        decision_info = "未知标签 '{}' 或颜色 '{}'".format(label, color_name)

    rospy.loginfo("发现最近的物体: {}, 颜色: {}".format(label, color_name))

    if target_zone:
        target_coords = TARGET_POSITIONS[target_zone]
        object_map_coords = closest_object['map_coords']
        
        current_state = 'pushing'  # 切换到推罐子状态
        rospy.loginfo("决策: {}。目标区域: {}。开始执行任务序列。".format(decision_info, target_zone))

        # --- 4. 执行导航序列 ---
        success = False
        try:
            rospy.loginfo("步骤 1: 导航到物体位置...")
            if move_to_goal(object_map_coords):
                rospy.loginfo("步骤 2: 推向目标区域...")
                if move_to_goal(target_coords):
                    success = True
                    processed_objects.add(closest_object['id'])
                else:
                    rospy.logwarn("推向目标区域失败，任务中断。")
            else:
                rospy.logwarn("导航到物体位置失败，任务中断。")
        finally:
            # 无论成功与否，都尝试返回观察点
            current_state = 'returning'  # 切换到返回状态
            rospy.loginfo("步骤 3: 返回观察点...")
            if move_to_goal(current_observation_pose, OBSERVATION_ORIENTATION):
                if success:
                    rospy.loginfo("任务完成，已返回观察点。")
                    # 更新观察点，为下一个物体做准备 (向右平移)
                    next_observation_pose = current_observation_pose.copy()
                    next_observation_pose[1] -= TRAVERSAL_STEP
                    current_observation_pose = next_observation_pose
                else:
                    rospy.logwarn("返回观察点成功，但任务失败。")
            else:
                rospy.logwarn("返回观察点失败！")
            current_state = 'cruising'  # 切换回巡航状态
    else:
        rospy.logwarn("{}，无对应操作。".format(decision_info))


def main():
    global tf_listener, move_base_client, current_state
    rospy.init_node('decision_maker_node', anonymous=True)

    tf_listener = tf.TransformListener()

    # 初始化move_base action客户端
    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("正在等待 move_base action 服务器...")
    move_base_client.wait_for_server()
    rospy.loginfo("move_base action 服务器已连接。")

    # 初始化状态
    current_state = 'cruising'

    # 启动巡航定时器，每5秒执行一次巡航
    cruise_timer = rospy.Timer(rospy.Duration(5.0), cruise)

    rospy.Subscriber('/detected_objects', String, objects_callback)

    rospy.loginfo("决策节点已启动，正在巡航并等待检测结果...")

    rospy.spin()

if __name__ == '__main__':
        main()