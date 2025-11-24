#!/usr/bin/env python
# coding: utf-8

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from aruco_msgs.msg import MarkerArray
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import tf
import numpy as np
from collections import deque
import threading
import math

# 导入 unitree_sdk2_python 的 SportClient 和 ChannelFactory
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.core.channel_factory import ChannelFactory


# --------------------------
# 配置参数
# --------------------------

# 二维码 ID 对应充电桩偏移量 (dx, dy, dz)
OFFSET_DICT = {
    0: (0.0, 0.0, -0.68),
    1: (0.25, -0.05, 0.0),
    2: (0.35, 0.05, 0.0)
}

# 导航目标发布话题
GOAL_TOPIC = "/move_base_simple/goal"

# 智能触发参数
STABILITY_FRAMES = 3          # 连续几帧位姿稳定才发布
PUBLISH_THRESHOLD = 0.05      # 发布目标点的最小位移变化（m）

# 下蹲动作参数
SQUAT_DURATION = 2.0           # 下蹲持续时间（秒）
SQUAT_SPEED = -0.1             # 对应机器人下蹲的速度映射


# --------------------------
# ChargingNavigator 类
# --------------------------
class ChargingNavigator:
    def __init__(self):
        rospy.init_node("charging_navigator")

        # --------------------------
        # 初始化 SportClient（用于控制机器人动作）
        # --------------------------
        self.sport_client = SportClient()
        self.sport_client.Init()  # 初始化连接
        self.sport_client.SetTimeout(10.0)  # 设置超时时间为 10 秒

        # 激活控制
        self.active = False
        self.goal_published = False

        # 订阅命令控制话题
        rospy.Subscriber("/aruco_cmd", String, self.cmd_callback)

        # 发布导航目标话题
        self.goal_pub = rospy.Publisher(GOAL_TOPIC, PoseStamped, queue_size=1)

        # 发布下蹲动作的控制话题（虽然现在不再用）
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # ArUco Marker 订阅
        rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, self.marker_callback)

        # TF 监听器，用于坐标转换
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 当前位置和目标点初始化
        self.current_pose = None
        self.final_goal = None
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # 启动检测线程（判断是否到达目标并触发下蹲动作）
        threading.Thread(target=self.check_goal_loop, daemon=True).start()

        rospy.loginfo("ChargingNavigator initialized. Waiting for /aruco_cmd to start recognition.")
        rospy.spin()


    def cmd_callback(self, msg):
        """
        控制二维码识别的启动和停止
        """
        if msg.data.lower() == "start":
            self.active = True
            self.goal_published = False  # 重置目标标志
            rospy.loginfo("Aruco recognition activated.")
        elif msg.data.lower() == "stop":
            self.active = False
            rospy.loginfo("Aruco recognition stopped.")


    def odom_callback(self, msg):
        """
        当前位置回调函数
        """
        self.current_pose = msg.pose.pose


    def marker_callback(self, msg):
        """
        ArUco Marker 识别回调函数，处理二维码识别结果
        """
        if not self.active or self.goal_published:
            return  # 未激活或已发布目标则忽略

        # 遍历检测到的所有 marker
        for marker in msg.markers:
            marker_id = marker.id
            if marker_id not in OFFSET_DICT:
                continue  # 如果 marker_id 不在我们的字典中，跳过

            dx, dy, dz = OFFSET_DICT[marker_id]
            marker_pose = marker.pose.pose

            # 计算二维码中心点加上偏移
            quat_marker = (
                marker_pose.orientation.x,
                marker_pose.orientation.y,
                marker_pose.orientation.z,
                marker_pose.orientation.w
            )
            rot_matrix = tf.transformations.quaternion_matrix(quat_marker)
            offset_vec = [dx, dy, dz, 1]
            charging_offset = rot_matrix.dot(offset_vec)

            charging_x = marker_pose.position.x + charging_offset[0]
            charging_y = marker_pose.position.y + charging_offset[1]
            charging_z = marker_pose.position.z + charging_offset[2]

            # 将二维码目标点转换为 map 坐标系下的坐标
            point_camera = PointStamped()
            point_camera.header.frame_id = "camera_link"
            point_camera.header.stamp = rospy.Time.now()
            point_camera.point.x = charging_x
            point_camera.point.y = charging_y
            point_camera.point.z = charging_z

            try:
                point_map = self.tf_buffer.transform(point_camera, "map", rospy.Duration(1.0))
            except Exception as e:
                rospy.logwarn("TF transform failed: %s", e)
                continue

            # 将二维码的姿态转换到 map 坐标系
            marker_in_cam = PoseStamped()
            marker_in_cam.header.frame_id = "camera_link"
            marker_in_cam.header.stamp = rospy.Time.now()
            marker_in_cam.pose = marker_pose

            try:
                marker_in_map = self.tf_buffer.transform(marker_in_cam, "map", rospy.Duration(1.0))
            except Exception as e:
                rospy.logwarn("TF pose transform failed: %s", e)
                continue

            # 提取二维码的朝向 yaw
            quat_marker_map = (
                marker_in_map.pose.orientation.x,
                marker_in_map.pose.orientation.y,
                marker_in_map.pose.orientation.z,
                marker_in_map.pose.orientation.w
            )
            R = tf.transformations.quaternion_matrix(quat_marker_map)[:3, :3]

            marker_z_map = R.dot([0.0, 0.0, 1.0])
            robot_forward_map = [-marker_z_map[0], -marker_z_map[1], 0.0]

            vx, vy = robot_forward_map[0], robot_forward_map[1]
            norm_xy = math.hypot(vx, vy)
            if norm_xy < 1e-6:
                _, _, yaw_marker = tf.transformations.euler_from_quaternion(quat_marker_map)
                yaw_robot = yaw_marker + math.pi
            else:
                yaw_robot = math.atan2(vy, vx)

            yaw_robot = math.atan2(math.sin(yaw_robot), math.cos(yaw_robot))
            quat_goal = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw_robot)

            # 发布目标点
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = rospy.Time.now()
            goal.pose.position.x = point_map.point.x
            goal.pose.position.y = point_map.point.y
            goal.pose.position.z = 0.0
            goal.pose.orientation.x = quat_goal[0]
            goal.pose.orientation.y = quat_goal[1]
            goal.pose.orientation.z = quat_goal[2]
            goal.pose.orientation.w = quat_goal[3]

            self.goal_pub.publish(goal)
            rospy.loginfo(f"Published goal (facing marker {marker_id})")

            self.goal_published = True
            self.final_goal = goal
            break


    def check_goal_loop(self):
        """
        检查是否到达目标，并触发下蹲动作
        """
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.final_goal is None or self.current_pose is None:
                rate.sleep()
                continue

            dx = self.final_goal.pose.position.x - self.current_pose.position.x
            dy = self.final_goal.pose.position.y - self.current_pose.position.y
            dist = math.hypot(dx, dy)

            # 如果到达目标点，执行下蹲动作
            if dist < 0.1:
                rospy.loginfo("Reached final goal, executing squat.")
                self.squat_motion()  # 使用 SportClient 执行下蹲动作
                self.final_goal = None

            rate.sleep()


    def squat_motion(self):
        """
        通过 SportClient 控制机器人下蹲
        """
        rospy.loginfo("Executing squat motion using SportClient...")
        self.sport_client.StandDown()  # 执行下趴动作
        #rospy.sleep(2)  # 下趴持续时间（秒）
        #self.sport_client.RiseSit()  # 恢复到站立
        #rospy.loginfo("Squat motion completed.")


# --------------------------
# 主程序入口
# --------------------------
if __name__ == "__main__":
    try:
        ChargingNavigator()  # 启动导航控制器
    except rospy.ROSInterruptException:
        pass

