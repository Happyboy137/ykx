#!/usr/bin/env python
# coding: utf-8

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped
from aruco_msgs.msg import MarkerArray
from std_msgs.msg import String
import tf
import math
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


# --------------------------
# 配置参数
# --------------------------

# 二维码 ID 对应充电桩偏移量 (dx, dy, dz)
OFFSET_DICT = {
    0: (0.0, 0.0, -1.0),
    1: (0.25, -0.05, 0.0),
    2: (0.35, 0.05, 0.0)
}

# move_base action server 名称
MOVE_BASE_ACTION_NAME = "move_base"

# 控制话题：到达后发布消息给控制端执行下蹲动作
CONTROL_TOPIC = "/charging_action"
CONTROL_MESSAGE = "arrived"  # 字符串内容，可由控制端解析执行动作

# TF 变换超时
TF_TIMEOUT = 2.0  # 秒


# --------------------------
# 主类：识别 ArUco 后导航到充电桩
# --------------------------
class ChargingNavigator:
    def __init__(self):
        rospy.init_node("charging_navigator")

        # --------------------------
        # TF 监听器
        # --------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --------------------------
        # 发布控制动作（到达触发下蹲）
        # --------------------------
        self.control_pub = rospy.Publisher(CONTROL_TOPIC, String, queue_size=1)

        # --------------------------
        # move_base Action Client
        # --------------------------
        self.move_client = actionlib.SimpleActionClient(MOVE_BASE_ACTION_NAME, MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        if not self.move_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server not available!")
            rospy.signal_shutdown("move_base not running")
            return
        rospy.loginfo("move_base action server is online.")

        # --------------------------
        # 当前目标状态
        # --------------------------
        self.goal_sent = False
        self.current_goal = None  # PoseStamped

        # --------------------------
        # 启动监听器
        # --------------------------
        self.start_listening = False  # 用于控制是否开始监听 ArUco 标记
        rospy.Subscriber("/goal_manager/return_complete", String, self.start_callback)

        rospy.loginfo("ChargingNavigator initialized. Waiting for start command...")
        rospy.spin()

    # --------------------------
    # 启动控制回调：接收到 'start' 命令后开始监听 ArUco 标记
    # --------------------------
    def start_callback(self, msg):
        if msg.data == "return_complete" and not self.start_listening:
            rospy.loginfo("Received 'start' command. Now listening for ArUco markers.")
            self.start_listening = True
            rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, self.marker_callback)

    # --------------------------
    # ArUco 回调
    # --------------------------
    def marker_callback(self, msg):
        # 如果当前已有目标正在导航，则忽略新 marker
        if self.goal_sent:
            return

        if not msg.markers:
            return

        # 只处理第一个匹配的 marker
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
            offset_vec = [dx, dy, dz, 1.0]
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
                point_map = self.tf_buffer.transform(point_camera, "map", rospy.Duration(TF_TIMEOUT))
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
                marker_in_map = self.tf_buffer.transform(marker_in_cam, "map", rospy.Duration(TF_TIMEOUT))
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

            # --------------------------
            # 封装 MoveBaseGoal 并发送给 move_base
            # --------------------------
            goal_msg = MoveBaseGoal()
            goal_msg.target_pose.header.frame_id = "map"
            goal_msg.target_pose.header.stamp = rospy.Time.now()
            goal_msg.target_pose.pose.position.x = point_map.point.x
            goal_msg.target_pose.pose.position.y = point_map.point.y
            goal_msg.target_pose.pose.position.z = 0.0
            goal_msg.target_pose.pose.orientation.x = quat_goal[0]
            goal_msg.target_pose.pose.orientation.y = quat_goal[1]
            goal_msg.target_pose.pose.orientation.z = quat_goal[2]
            goal_msg.target_pose.pose.orientation.w = quat_goal[3]

            # 发送 goal 并指定 done_cb（到达/失败回调）
            self.move_client.send_goal(goal_msg, done_cb=self._move_done_cb)

            rospy.loginfo(f"Sent MoveBaseGoal for marker {marker_id} at "
                          f"({point_map.point.x:.2f}, {point_map.point.y:.2f}), yaw={yaw_robot:.2f} rad")

            self.goal_sent = True
            self.current_goal = goal_msg
            break

    # --------------------------
    # move_base done callback
    # --------------------------
    def _move_done_cb(self, status, result):
        """
        当 move_base 完成导航时调用
        status:
            3 == SUCCEEDED
            4 == ABORTED
            5 == REJECTED/PREEMPTED
        """
        if status == 3:  # SUCCEEDED
            rospy.loginfo("Navigation succeeded. Publishing control message to trigger squat.")
            self.control_pub.publish(String(CONTROL_MESSAGE))
        else:
            rospy.logwarn(f"Navigation failed with status {status}.")

        # 重置状态，以便下一个 marker 可以触发导航
        self.goal_sent = False
        self.current_goal = None


# --------------------------
# 主程序入口
# --------------------------
if __name__ == "__main__":
    try:
        ChargingNavigator()
    except rospy.ROSInterruptException:
        pass

