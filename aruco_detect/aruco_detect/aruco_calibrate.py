#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.task import Future
import cv2
import numpy as np
import itertools
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import os
from collections import defaultdict

from . import utils

ARUCO_DICT = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_5X5_1000": aruco.DICT_5X5_1000,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_6X6_1000": aruco.DICT_6X6_1000,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL
}

def get_aruco_dict(marker_type_name: str):
    code = ARUCO_DICT[marker_type_name]
    if hasattr(aruco, 'getPredefinedDictionary'):      # OpenCV ≥ 4.7
        return aruco.getPredefinedDictionary(code)
    else:                                               # OpenCV ≤ 4.6
        return aruco.Dictionary_get(code)

class ArucoCalibrate(Node):
    def __init__(self):
        super().__init__('aruco_marker_calibration')

        self.declare_parameter('aruco_type', 'DICT_6X6_100')
        self.declare_parameter('aruco_length', 0.0489)
        self.declare_parameter('aruco_save_dir', '.')
        self.declare_parameter('camera_img_topic', '/camera/rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/rgb/camera_info')
        self.declare_parameter('camera_frame_id', 'rgb_camera_link')
        self.declare_parameter('aruco_main_marker_id', 0)
        self.declare_parameter('calibration_duration', 60.0)

        self.marker_type = self.get_parameter('aruco_type').get_parameter_value().string_value
        self.marker_size = self.get_parameter('aruco_length').get_parameter_value().double_value
        self.save_dir = self.get_parameter('aruco_save_dir').get_parameter_value().string_value
        self.camera_img_topic = self.get_parameter('camera_img_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_frame_id = self.get_parameter('camera_frame_id').get_parameter_value().string_value
        self.main_marker_id = self.get_parameter('aruco_main_marker_id').get_parameter_value().integer_value
        self.calibration_duration = self.get_parameter('calibration_duration').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.marker_transforms_list = []
        self.marker_id_list = []
        self.marker_updates_list = []
        self.marker_pose_list = PoseArray()
        self.detected_ids = []

        self.aruco_pub = self.create_publisher(Image, "aruco_img", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.image_sub = self.create_subscription(Image, self.camera_img_topic, self.img_cb, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_cb, 10)

        self.K = None
        self.D = None

        # 종료 시그널용 Future
        self.done_fut: Future = Future()

        self.get_logger().info(f"Calibration starting. Make sure to correctly set the Save Directory: {self.save_dir}")
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        if self.K is None:
            return

        duration = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if duration < self.calibration_duration:
            self.find_transforms()
            self.get_logger().info(
                f"Calibrating... {int(self.calibration_duration - duration)}s left. Detected IDs: {self.detected_ids}"
            )
        else:
            # 여기서는 파일 저장까지만 하고, 종료는 main에서 처리
            self.set_transforms(self.main_marker_id)
            self.get_logger().info("Calibration finished successfully.")
            if not self.done_fut.done():
                self.done_fut.set_result(True)

    def img_cb(self, msg):
        try:
            color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.K is not None:
                _, self.marker_pose_list, self.detected_ids = self.detect_aruco(color_img)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def info_cb(self, msg):
        self.K = np.reshape(msg.k, (3, 3))
        self.D = np.array(msg.d)
        # 구독 해제는 Node API로 처리하고 참조 정리
        self.destroy_subscription(self.info_sub)
        self.info_sub = None

    def detect_aruco(self, img, broadcast_markers_tf=True):
        aruco_dict = get_aruco_dict(self.marker_type)

        # OpenCV 버전에 따라 분기
        new_api = hasattr(aruco, 'ArucoDetector') and hasattr(aruco, 'DetectorParameters')
        if new_api:
            parameters = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(img)
        else:
            parameters = aruco.DetectorParameters_create()  # ← 구버전 API
            corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

        marker_pose_list = PoseArray()
        id_list = []
        output_img = img.copy()

        if ids is not None and len(ids) > 0:
            # pose 추정은 양쪽 버전 모두 동일 함수 사용
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.K, self.D)
            for i, marker_id in enumerate(ids):
                rvec, tvec = rvecs[i], tvecs[i]

                # 디텍션 결과 그리기 (API 동일)
                aruco.drawDetectedMarkers(output_img, [corners[i]], np.array([marker_id]))
                cv2.drawFrameAxes(output_img, self.K, self.D, rvec, tvec, 0.03)

                marker_pose = self.make_pose(rvec, tvec)

                if broadcast_markers_tf:
                    tf_marker = TransformStamped()
                    tf_marker.header.stamp = self.get_clock().now().to_msg()
                    tf_marker.header.frame_id = self.camera_frame_id
                    tf_marker.child_frame_id = f"marker_{int(marker_id[0])}"
                    tf_marker.transform.translation.x = marker_pose.position.x
                    tf_marker.transform.translation.y = marker_pose.position.y
                    tf_marker.transform.translation.z = marker_pose.position.z
                    tf_marker.transform.rotation = marker_pose.orientation
                    self.tf_broadcaster.sendTransform(tf_marker)

                marker_pose_list.poses.append(marker_pose)
                id_list.append(int(marker_id))

        out_img_msg = self.bridge.cv2_to_imgmsg(output_img, "bgr8")
        self.aruco_pub.publish(out_img_msg)
        return output_img, marker_pose_list, id_list


    def make_pose(self, rvec, tvec):
        marker_pose = Pose()
        rot = R.from_rotvec(np.squeeze(rvec))
        quat = rot.as_quat()
        tvec_sq = np.squeeze(tvec)
        marker_pose.position.x = float(tvec_sq[0])
        marker_pose.position.y = float(tvec_sq[1])
        marker_pose.position.z = float(tvec_sq[2])
        marker_pose.orientation.x = float(quat[0])
        marker_pose.orientation.y = float(quat[1])
        marker_pose.orientation.z = float(quat[2])
        marker_pose.orientation.w = float(quat[3])
        return marker_pose

    def find_transforms(self):
        id_index = range(len(self.detected_ids))
        pose_combinations = list(itertools.combinations(id_index, 2))

        for i, j in pose_combinations:
            combination = [self.detected_ids[i], self.detected_ids[j]]
            pose_0 = self.marker_pose_list.poses[i]
            pose_1 = self.marker_pose_list.poses[j]

            tf_matrix_0 = utils.pose_to_matrix(pose_0)
            tf_matrix_1 = utils.pose_to_matrix(pose_1)
            tf_0_to_1 = np.dot(np.linalg.inv(tf_matrix_0), tf_matrix_1)
            trans, rotation = utils.matrix_to_quat_trans(tf_0_to_1)

            if combination not in self.marker_id_list and combination[::-1] not in self.marker_id_list:
                self.marker_transforms_list.append([np.array(trans), np.array(rotation)])
                self.marker_id_list.append(combination)
                self.marker_updates_list.append(1)
            else:
                try:
                    combination_idx = self.marker_id_list.index(combination)
                except ValueError:
                    combination_idx = self.marker_id_list.index(combination[::-1])
                    trans, rotation = utils.matrix_to_quat_trans(np.linalg.inv(tf_0_to_1))

                average_translation = 0.99 * self.marker_transforms_list[combination_idx][0] + 0.01 * np.array(trans)
                average_rotation = utils.average_quaternions(
                    [self.marker_transforms_list[combination_idx][1], np.array(rotation)],
                    weights=[0.9, 0.1]
                )
                self.marker_transforms_list[combination_idx][0] = average_translation
                self.marker_transforms_list[combination_idx][1] = average_rotation
                self.marker_updates_list[combination_idx] += 1

    def set_transforms(self, id_main):
        graph = self.build_graph(self.marker_id_list)
        paths = {}
        mk_tf = {}
        for start in graph.keys():
            if start == id_main:
                continue
            paths[start] = self.bfs_sp(graph, start, id_main)

        for marker_id, path in paths.items():
            if path is None:
                continue

            final_transform = np.identity(4)
            for i in range(len(path) - 1):
                comb = [path[i], path[i+1]]
                try:
                    idx = self.marker_id_list.index(comb)
                    trans, rot = self.marker_transforms_list[idx]
                    mat = utils.quat_trans_to_matrix(trans, rot)
                except ValueError:
                    idx = self.marker_id_list.index(comb[::-1])
                    trans, rot = self.marker_transforms_list[idx]
                    mat = np.linalg.inv(utils.quat_trans_to_matrix(trans, rot))
                final_transform = np.dot(final_transform, mat)
            mk_tf[marker_id] = final_transform

        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, 'marker_transforms.npz')
        np.savez(save_path, mk_tf_dict=mk_tf)
        self.get_logger().info(f"Transforms saved to {save_path}")

    def build_graph(self, edges):
        graph = defaultdict(list)
        for edge in edges:
            a, b = edge[0], edge[1]
            graph[a].append(b)
            graph[b].append(a)
        return graph

    def bfs_sp(self, graph, start, goal):
        explored = []
        queue = [[start]]
        if start == goal:
            return [start]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                neighbours = graph[node]
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    if neighbour == goal:
                        return new_path
                explored.append(node)
        self.get_logger().warning(f"Connecting path from {start} to {goal} doesn't exist!")
        return None

def main(args=None):
    rclpy.init(args=args)
    node = ArucoCalibrate()
    try:
        # 노드가 done_fut를 완료시키면 정상적으로 spin 종료
        rclpy.spin_until_future_complete(node, node.done_fut)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
