#!/usr/bin/env python
# coding:utf-8

import serial
import time
import sys
import math
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
import tf

strat_0 = False
strat_1 = False
count = 0
received_data = []
LENTH = 0.157
TRACK_WIDTH = 0.219

speed = 0.0
w = 0.0
roll = 0.0

# 里程计变量
x = 0.0
y = 0.0
theta = 0.0
last_time = time.time()

# 配置串口参数
try:
    ser = serial.Serial(
        port='/dev/ttyUSB0',    # 你的串口设备路径
        baudrate=115200,          # 波特率，需与设备匹配
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1               # 读超时设置为1秒
    )
except serial.SerialException as e:
    print("串口错误")
    sys.exit(1)

# ROS 初始化
rospy.init_node('odometry_publisher')
odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
tf_broadcaster = tf.TransformBroadcaster()

try:
    # 检查串口是否成功打开
    if ser.isOpen():
        print("成功打开串口")

    # 循环读取数据
    while not rospy.is_shutdown():
        # 检查接收缓冲区中是否有数据
        if ser.inWaiting() > 0:
            # 以字节为单位读取数据
            # 读取单个字节
            single_byte = ser.read(1)
            if single_byte == b'\xff':
                strat_0 = True
            elif single_byte == b'\xfe' and strat_0 == True:
                strat_1 = True
                count = 0
            elif strat_1 == True and strat_0 == True:
                count += 1
                received_data.append(ord(single_byte))
            
            if count == 8:
                strat_0 = False
                strat_1 = False
                count = 0
                encoder_left = received_data[0] * (1.0 - received_data[1])
                encoder_right = received_data[2] * (1.0 - received_data[3])
                roll = float((received_data[4] * 256 + received_data[5]) - 15000) / 100
                gz = (received_data[6] * 256 + received_data[7] - 32768) / 16.4
                
                speed = 0.5 * (encoder_left + encoder_right) / 15.6 * LENTH
                w = gz / 180 * math.pi  # 转换为弧度
                
                # 更新里程计
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # 积分更新位置
                theta += w * dt
                x += speed * math.cos(theta) * dt
                y += speed * math.sin(theta) * dt
                
                # 发布 Odometry
                odom = Odometry()
                odom.header.stamp = rospy.Time.now()
                odom.header.frame_id = "odom"
                odom.child_frame_id = "base_link"
                
                # 位置
                odom.pose.pose.position.x = x
                odom.pose.pose.position.y = y
                odom.pose.pose.position.z = 0.0
                quat = tf.transformations.quaternion_from_euler(0, 0, theta)
                # 设置姿态
                odom.pose.pose.orientation = Quaternion(*quat)
                
                # 速度
                odom.twist.twist.linear.x = speed
                odom.twist.twist.angular.z = w
                
                # 协方差（简化）
                odom.pose.covariance = [0.1, 0, 0, 0, 0, 0,
                                        0, 0.1, 0, 0, 0, 0,
                                        0, 0, 0.1, 0, 0, 0,
                                        0, 0, 0, 0.1, 0, 0,
                                        0, 0, 0, 0, 0.1, 0,
                                        0, 0, 0, 0, 0, 0.1]
                odom.twist.covariance = odom.pose.covariance
                
                odom_pub.publish(odom)
                
                # 广播 TF
                tf_broadcaster.sendTransform(
                    (x, y, 0),
                    quat,
                    odom.header.stamp,
                    "base_link",
                    "odom"
                )
                
                received_data = []  # 清空数据

        # 短暂睡眠，避免CPU占用过高
        time.sleep(0.01) 

except serial.SerialException as e:
    print("串口错误")
except KeyboardInterrupt:
    print("程序被用户中断")
finally:
    # 确保串口最终被关闭
    if ser and ser.isOpen():
        ser.close()
        print("串口已关闭")