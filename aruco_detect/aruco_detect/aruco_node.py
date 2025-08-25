#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import os

from . import utils




# Names of each possible ArUco tag OpenCV supports
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

class ArucoNode(Node):
    def __init__(self):
        super().__init__('aruco_marker_detect')

        # Declare parameters
        self.declare_parameter('aruco_type', 'DICT_6X6_100')
        self.declare_parameter('aruco_length', 0.0489)
        self.declare_parameter('aruco_transforms', '')
        self.declare_parameter('aruco_update_rate', 0.1)
        self.declare_parameter('aruco_main_marker_id', 0)
        self.declare_parameter('aruco_obj_id', 'aruco_obj')
        self.declare_parameter('camera_img_topic', '/camera/rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/rgb/camera_info')
        self.declare_parameter('camera_frame_id', 'rgb_camera_link')

        # Get parameters
        self.marker_type = self.get_parameter('aruco_type').get_parameter_value().string_value
        self.marker_size = self.get_parameter('aruco_length').get_parameter_value().double_value
        self.marker_transform_file = self.get_parameter('aruco_transforms').get_parameter_value().string_value
        self.aruco_update_rate = self.get_parameter('aruco_update_rate').get_parameter_value().double_value
        self.aruco_main_marker_id = self.get_parameter('aruco_main_marker_id').get_parameter_value().integer_value
        self.aruco_obj_id = self.get_parameter('aruco_obj_id').get_parameter_value().string_value
        self.camera_img_topic = self.get_parameter('camera_img_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_frame_id = self.get_parameter('camera_frame_id').get_parameter_value().string_value

        self.bridge = CvBridge()
        
        #---- Markers detected at each camera frame ----#
        self.marker_pose_list = PoseArray()
        self.detected_ids = []
        #----------------------------------------------#

        #---- Used at prediction time ----#
        self.obj_transform = Pose()

        if not self.marker_transform_file:
            self.get_logger().error("No marker transforms provided. Shutting Down")
            raise ValueError("No marker transforms provided.")
        
        try:
            self.marker_transforms = self.load_marker_transform(self.marker_transform_file)
        except Exception as e:
            self.get_logger().error(f"Invalid marker transform file: {e}")
            raise

        # ROS Publisher
        self.aruco_pub = self.create_publisher(Image, "aruco_img", 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # ROS Subscriber
        self.image_sub = self.create_subscription(Image, self.camera_img_topic, self.img_cb, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_cb, 10)

        self.K = None
        self.D = None

        # Timer for the main logic
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.get_logger().info("ArUco node has been started.")

    def timer_callback(self):
        if self.K is not None and len(self.detected_ids) > 0:
            self.calculate_transform(self.aruco_main_marker_id)

    def load_marker_transform(self, marker_transform_file):
        if not os.path.exists(marker_transform_file):
             # Try to resolve path relative to package share directory
            package_share_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            marker_transform_file = os.path.join(package_share_directory, marker_transform_file)

        if not os.path.exists(marker_transform_file):
            raise FileNotFoundError(f"Marker transform file not found at {marker_transform_file}")

        load_unformated = np.load(marker_transform_file, allow_pickle=True)
        mk_transform = load_unformated['mk_tf_dict'][()]
        self.get_logger().info("TF between markers successfully loaded from file.")
        return mk_transform

    def img_cb(self, msg):
        try:
            color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.K is not None:
                markers_img, marker_pose_list, id_list = self.detect_aruco(color_img)
                self.marker_pose_list = marker_pose_list
                self.detected_ids = id_list
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def info_cb(self, msg):
        self.K = np.reshape(msg.k, (3, 3))
        self.D = np.array(msg.d)
        # self.info_sub.destroy() # Unsubscribe after getting camera info
        self.destroy_subscription(self.info_sub)
        self.info_sub = None

    def _get_dict(self, code):
        # OpenCV ≥ 4.7
        if hasattr(aruco, 'getPredefinedDictionary'):
            return aruco.getPredefinedDictionary(code)
        # OpenCV ≤ 4.6
        return aruco.Dictionary_get(code)

    def detect_aruco(self, img, broadcast_markers_tf=False):
        aruco_dict = self._get_dict(ARUCO_DICT[self.marker_type])

        # 신/구 API 분기
        new_api = hasattr(aruco, 'ArucoDetector') and hasattr(aruco, 'DetectorParameters')
        if new_api:
            params = aruco.DetectorParameters()
            params.minCornerDistanceRate = 0.02
            params.minMarkerDistanceRate = 0.02
            params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
            detector = aruco.ArucoDetector(aruco_dict, params)
            corners, ids, rejected = detector.detectMarkers(img)
        else:
            params = aruco.DetectorParameters_create()
            params.minCornerDistanceRate = 0.02
            params.minMarkerDistanceRate = 0.02
            params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
            corners, ids, rejected = aruco.detectMarkers(img, aruco_dict, parameters=params)

        marker_pose_list = PoseArray()
        id_list = []
        output_img = img.copy()

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.K, self.D)
            for i, marker_id in enumerate(ids):
                rvec, tvec = rvecs[i], tvecs[i]
                aruco.drawDetectedMarkers(output_img, [corners[i]], np.array([marker_id]))
                cv2.drawFrameAxes(output_img, self.K, self.D, rvec, tvec, 0.05)

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
        tvec = np.squeeze(tvec)
        rvec = np.squeeze(rvec)
        
        rot = R.from_rotvec(rvec)
        quat = rot.as_quat()

        marker_pose.position.x = float(tvec[0])
        marker_pose.position.y = float(tvec[1])
        marker_pose.position.z = float(tvec[2])
        marker_pose.orientation.x = float(quat[0])
        marker_pose.orientation.y = float(quat[1])
        marker_pose.orientation.z = float(quat[2])
        marker_pose.orientation.w = float(quat[3])
        return marker_pose

    def calculate_transform(self, id_main):
        marker_pose_list, detected_ids = self.marker_pose_list, self.detected_ids
        transforms_rot = []
        transforms_trans = []

        for i, marker_id in enumerate(detected_ids):
            trans, rot = utils.pose_to_quat_trans(marker_pose_list.poses[i])
            
            if marker_id == id_main:
                transforms_rot.append(rot)
                transforms_trans.append(trans)
                continue
            else:
                if marker_id not in self.marker_transforms:
                    self.get_logger().warning(f"Unknown marker ID detected {marker_id}")
                    continue
                
                tf_matrix = utils.quat_trans_to_matrix(trans, rot)
                full_tf = np.dot(tf_matrix, self.marker_transforms[marker_id])
                trans_new, rot_new = utils.matrix_to_quat_trans(full_tf)
                transforms_rot.append(rot_new)
                transforms_trans.append(trans_new)

        if not transforms_rot:
            return

        avg_rot = utils.average_quaternions(np.array(transforms_rot))
        avg_trans = np.mean(np.array(transforms_trans), axis=0)

        object_tf = TransformStamped()
        object_tf.header.stamp = self.get_clock().now().to_msg()
        object_tf.header.frame_id = self.camera_frame_id
        object_tf.child_frame_id = self.aruco_obj_id
        
        pose = utils.quat_trans_to_pose(avg_trans, avg_rot)
        object_tf.transform.translation.x = pose.position.x
        object_tf.transform.translation.y = pose.position.y
        object_tf.transform.translation.z = pose.position.z
        object_tf.transform.rotation = pose.orientation
        
        self.tf_broadcaster.sendTransform(object_tf)

def main(args=None):
    rclpy.init(args=args)
    aruco_node = ArucoNode()
    try:
        rclpy.spin(aruco_node)
    except KeyboardInterrupt:
        pass
    finally:
        aruco_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
