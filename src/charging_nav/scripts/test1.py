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
# 下蹲动作函数
# --------------------------
def squat_motion(cmd_pub):
    """
    通过 /cmd_vel 发送下蹲动作
    """
    squat_cmd = Twist()
    squat_cmd.linear.z = SQUAT_SPEED  # 下蹲速度
    start_time = rospy.Time.now().to_sec()
    rate = rospy.Rate(10)  # 10 Hz 发布
    while rospy.Time.now().to_sec() - start_time < SQUAT_DURATION:
        cmd_pub.publish(squat_cmd)
        rate.sleep()
    # 下蹲完成，停止动作
    cmd_pub.publish(Twist())
    rospy.loginfo("Squat motion completed.")


# --------------------------
# 主类
# --------------------------
class ChargingNavigator:
    def __init__(self):
        rospy.init_node("charging_navigator")

        # --------------------------
        # 激活控制
        # --------------------------
        self.active = False  # 是否激活二维码识别
        self.goal_published = False  # 是否已经发布过目标
        rospy.Subscriber("/aruco_cmd", String, self.cmd_callback)

        # --------------------------
        # 发布导航目标
        # --------------------------
        self.goal_pub = rospy.Publisher(GOAL_TOPIC, PoseStamped, queue_size=1)

        # --------------------------
        # 发布下蹲动作
        # --------------------------
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # --------------------------
        # ArUco 订阅
        # --------------------------
        rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, self.marker_callback)

        # --------------------------
        # TF 监听器
        # --------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --------------------------
        # 当前位置和最终目标
        # --------------------------
        self.current_pose = None
        self.final_goal = None  # 当前目标点，用于触发下蹲
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # --------------------------
        # 启动检测线程（判断是否到达目标并下蹲）
        # --------------------------
        threading.Thread(target=self.check_goal_loop, daemon=True).start()

        rospy.loginfo("ChargingNavigator initialized. Waiting for /aruco_cmd to start recognition.")
        rospy.spin()


    # --------------------------
    # /aruco_cmd 回调
    # --------------------------
    def cmd_callback(self, msg):
        """
        接收 start/stop 控制信号
        """
        if msg.data.lower() == "start":
            self.active = True
            self.goal_published = False  # 重置
            rospy.loginfo("Aruco recognition activated.")
        elif msg.data.lower() == "stop":
            self.active = False
            rospy.loginfo("Aruco recognition stopped.")


    # --------------------------
    # 当前位置回调
    # --------------------------
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose


    # --------------------------
    # ArUco 回调
    # --------------------------
    def marker_callback(self, msg):
        if not self.active or self.goal_published:
            return  # 未激活或已发布目标则忽略

        for marker in msg.markers:
            marker_id = marker.id
            if marker_id not in OFFSET_DICT:
                continue

            dx, dy, dz = OFFSET_DICT[marker_id]
            marker_pose = marker.pose.pose

            # --------------------------
            # 将二维码中心点加偏移
            # --------------------------
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

            # --------------------------
            # 转换目标点到 map 坐标系
            # --------------------------
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

            # --------------------------
            # 转换二维码姿态到 map 坐标系
            # --------------------------
            marker_in_cam = PoseStamped()
            marker_in_cam.header.frame_id = "camera_link"
            marker_in_cam.header.stamp = rospy.Time.now()
            marker_in_cam.pose = marker_pose

            try:
                marker_in_map = self.tf_buffer.transform(marker_in_cam, "map", rospy.Duration(1.0))
            except Exception as e:
                rospy.logwarn("TF pose transform failed: %s", e)
                continue

            # --------------------------
            # 提取二维码朝向 yaw
            # --------------------------
            quat_marker_map = (
                marker_in_map.pose.orientation.x,
                marker_in_map.pose.orientation.y,
                marker_in_map.pose.orientation.z,
                marker_in_map.pose.orientation.w
            )
            R = tf.transformations.quaternion_matrix(quat_marker_map)[:3, :3]  # 3x3
            
            # 2) marker 的局部 +z 轴是 [0,0,1]，把它旋转到 map 下
            marker_z_map = R.dot([0.0, 0.0, 1.0])  # 3-vector
            # 3) 目标机器人前向应为 marker_z_map 的反向（指向二维码平面）
            robot_forward_map = [-marker_z_map[0], -marker_z_map[1], 0.0]  # 只取 XY 平面分量（扔掉z）
            
            # 4) 如果投影太小（marker 极度朝上/下），退回到直接用 yaw_from_quat 作为兜底
            vx, vy = robot_forward_map[0], robot_forward_map[1]
            norm_xy = math.hypot(vx, vy)
            if norm_xy < 1e-6:
            	# 兜底：使用 marker 在 map 下的 yaw（但通常不会发生）
            	_, _, yaw_marker = tf.transformations.euler_from_quaternion(quat_marker_map)
            	yaw_robot = yaw_marker + math.pi
            else:
            # 5) 由前向向量计算 yaw
            	yaw_robot = math.atan2(vy, vx)
            
            yaw_robot = math.atan2(math.sin(yaw_robot), math.cos(yaw_robot)) # 归一化到 [-pi, pi]
            quat_goal = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw_robot)

            # --------------------------
            # 发布导航目标
            # --------------------------
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
            rospy.loginfo(f"Published goal (facing marker {marker_id}) at ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}), yaw={yaw_robot:.2f} rad")

            # --------------------------
            # 标记为已发布
            # --------------------------
            #self.goal_published = True
            #self.final_goal = goal
            #break


    # --------------------------
    # 线程：检查是否到达目标并触发下蹲
    # --------------------------
    def check_goal_loop(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self.final_goal is None or self.current_pose is None:
                rate.sleep()
                continue

            dx = self.final_goal.pose.position.x - self.current_pose.position.x
            dy = self.final_goal.pose.position.y - self.current_pose.position.y
            dist = math.hypot(dx, dy)

            if dist < 0.1:
                rospy.loginfo("Reached final goal, executing squat")
                squat_motion(self.cmd_pub)
                self.final_goal = None  # 防止重复触发

            rate.sleep()


# --------------------------
# 主程序入口
# --------------------------
if __name__ == "__main__":
    try:
        ChargingNavigator()
    except rospy.ROSInterruptException:
        pass

