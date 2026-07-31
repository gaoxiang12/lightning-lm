#!/bin/bash
# 同步底盘时间并启动 RViz
# 用法: bash start_rviz.sh

set -e

echo "[1/2] 同步底盘时间..."
ssh sunrise@6.6.7.6 "sudo date -s '$(date)'" 2>/dev/null && echo "  底盘时间已同步: $(date)"

echo "[2/2] 启动 RViz..."
cd /home/tjzn/Workspace
source /opt/ros/humble/setup.bash
source install/setup.bash


rviz2 -d /home/tjzn/Workspace/src/lightning_nav2/rviz/navigation.rviz
