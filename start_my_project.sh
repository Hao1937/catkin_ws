#!/bin/bash

# =================================================================
#         Astra 相机项目一键启动脚本 (根据您的需求定制)
# =================================================================

# --- 1. 环境配置 (如果您的工作区不在主目录，请修改) ---

# 您的 ROS 发行版名称 (例如: melodic, noetic)
ROS_DISTRO="melodic"

# 您的 Catkin 工作空间的绝对路径
# "$HOME" 会自动替换为您的主目录, 例如 /home/your_username
CATKIN_WS_PATH="$HOME/catkin_ws"


# --- 2. 脚本执行区 (已根据您的命令填写) ---

echo "🚀  正在按顺序启动所有 ROS 节点..."

# 准备需要在新终端中执行的 source 命令
# 注意: 我已将您写的 dev/setup.bash 修正为正确的 devel/setup.bash
SOURCE_CMD="source /opt/ros/$ROS_DISTRO/setup.bash; source $CATKIN_WS_PATH/devel/setup.bash"

# --- 命令 1: 启动 roscore ---
TITLE_1="roscore"
COMMAND_1="roscore"
echo "1. 启动: $COMMAND_1"
gnome-terminal --tab --title="$TITLE_1" -- bash -c "$SOURCE_CMD; $COMMAND_1; exec bash"
sleep 3 # 等待 roscore 完全启动

# --- 命令 2: 启动第一个 astra launch 文件 ---
TITLE_2="Astra Camera"
COMMAND_2="roslaunch astra_camera astra.launch"
echo "2. 启动: $COMMAND_2"
gnome-terminal --tab --title="$TITLE_2" -- bash -c "$SOURCE_CMD; $COMMAND_2; exec bash"
sleep 5 # 等待相机节点启动，可以适当延长

# --- 命令 3: 启动第二个 astra launch 文件 ---
TITLE_3="Astra Stereo"
COMMAND_3="roslaunch astra_camera stereo_s.launch"
echo "3. 启动: $COMMAND_3"
gnome-terminal --tab --title="$TITLE_3" -- bash -c "$SOURCE_CMD; $COMMAND_3; exec bash"
sleep 2

# --- 命令 4: 运行您的 Python 脚本 ---
# 注意: 这里的命令会先 cd 到工作区目录，再执行 python 脚本
TITLE_4="Python Script"
COMMAND_4="cd $CATKIN_WS_PATH/src/colordetector/scripts; python3 img2.py"
echo "4. 启动: $COMMAND_4"
gnome-terminal --tab --title="$TITLE_4" -- bash -c "$SOURCE_CMD; $COMMAND_4; exec python3"
sleep 2

'''
# --- 命令 5: 运行您的 decisionmaker 脚本 ---
# 注意: 这里的命令会先 cd 到工作区目录，再执行 python 脚本
TITLE_4="Python Script"
COMMAND_4="cd $CATKIN_WS_PATH/src/colordetector/scripts; rosrun colordetector decision_maker.py"
echo "4. 启动: $COMMAND_4"
gnome-terminal --tab --title="$TITLE_4" -- bash -c "$SOURCE_CMD; $COMMAND_4; exec bash"
sleep 2
'''

echo "✅  所有命令已在新的终端标签页中启动！"