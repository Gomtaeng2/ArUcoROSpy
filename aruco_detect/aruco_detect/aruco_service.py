#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import os

from aruco_interfaces.srv import ArucoPoseEstimate
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

class ArucoService(Node):
    def __init__(self):
        super().__init__('aruco_pose_service_node')

        self.declare_parameter('aruco_type', 'DICT_6X6_100')
        self.declare_parameter('aruco_length', 0.0489)
        self.declare_parameter('aruco_transforms', '')
        self.declare_parameter('aruco_main_marker_id', 0)

        self.marker_type = self.get_parameter('aruco_type').get_parameter_value().string_value
        self.marker_size = self.get_parameter('aruco_length').get_parameter_value().double_value
        self.marker_transform_file = self.get_parameter('aruco_transforms').get_parameter_value().string_value
        self.main_marker_id = self.get_parameter('aruco_main_marker_id').get_parameter_value().integer_value

        self.bridge = CvBridge()
        
        if not self.marker_transform_file:
            self.get_logger().error("No marker transforms provided. Shutting Down")
            raise ValueError("No marker transforms provided.")
        
        try:
            self.marker_transforms = self.load_marker_transform(self.marker_transform_file)
        except Exception as e:
            self.get_logger().error(f"Invalid marker transform file: {e}")
            raise

        self.pose_estimate_srv = self.create_service(ArucoPoseEstimate, 'aruco_pose_estimate', self.estimate_pose_cb)
        self.aruco_pub = self.create_publisher(Image, "aruco_img", 10)
        self.get_logger().info("Aruco detection service ready.")

    def load_marker_transform(self, marker_transform_file):
        if not os.path.exists(marker_transform_file):
            package_share_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            marker_transform_file = os.path.join(package_share_directory, marker_transform_file)

        if not os.path.exists(marker_transform_file):
            raise FileNotFoundError(f"Marker transform file not found at {marker_transform_file}")

        load_unformated = np.load(marker_transform_file, allow_pickle=True)
        mk_transform = load_unformated['mk_tf_dict'][()]
        self.get_logger().info("TF between markers successfully loaded from file.")
        return mk_transform

    def estimate_pose_cb(self, request, response):
        image = request.image
        camera_info = request.camera_info
        K, D = self.caminfo_to_matrx_dist(camera_info)

        try:
            color_img = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            response.success = False
            return response

        output_img, marker_pose_list, detected_id_list = self.detect_aruco(color_img, K, D)
        estimated_pose = self.calculate_transform(self.main_marker_id, marker_pose_list, detected_id_list)

        if estimated_pose is None:
            response.success = False
        else:
            response.success = True
            response.pose = estimated_pose
        return response

    def caminfo_to_matrx_dist(self, caminfo):
        K = np.reshape(caminfo.k, (3, 3))
        D = np.array(caminfo.d)
        return K, D
    
    def _get_dict(self, code):
        # OpenCV ≥ 4.7
        if hasattr(aruco, 'getPredefinedDictionary'):
            return aruco.getPredefinedDictionary(code)
        # OpenCV ≤ 4.6
        return aruco.Dictionary_get(code)

    def detect_aruco(self, img, camera_matrix, dist_coeffs):
        aruco_dict = self._get_dict(ARUCO_DICT[self.marker_type])

        # 신/구 API 감지
        new_api = hasattr(aruco, 'ArucoDetector') and hasattr(aruco, 'DetectorParameters')
        if new_api:
            params = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(img)
        else:
            params = aruco.DetectorParameters_create()
            corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=params)

        marker_pose_list = []
        id_list = []
        output_img = img.copy()

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, camera_matrix, dist_coeffs)
            for i, marker_id in enumerate(ids):
                aruco.drawDetectedMarkers(output_img, [corners[i]], np.array([marker_id]))
                cv2.drawFrameAxes(output_img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)
                marker_pose = self.make_pose(rvecs[i], tvecs[i])
                marker_pose_list.append(marker_pose)
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

    def calculate_transform(self, id_main, marker_pose_list, detected_ids):
        transforms_rot = []
        transforms_trans = []
        for i, marker_id in enumerate(detected_ids):
            trans, rot = utils.pose_to_quat_trans(marker_pose_list[i])
            if marker_id == id_main:
                transforms_rot.append(rot)
                transforms_trans.append(trans)
            elif marker_id in self.marker_transforms:
                tf_matrix = utils.quat_trans_to_matrix(trans, rot)
                full_tf = np.dot(tf_matrix, self.marker_transforms[marker_id])
                trans_new, rot_new = utils.matrix_to_quat_trans(full_tf)
                transforms_rot.append(rot_new)
                transforms_trans.append(trans_new)
            else:
                self.get_logger().warning(f"Unknown Aruco marker present: {marker_id}")

        if not transforms_rot:
            return None

        avg_rot = utils.average_quaternions(np.array(transforms_rot))
        avg_trans = np.mean(np.array(transforms_trans), axis=0)
        return utils.quat_trans_to_pose(avg_trans, avg_rot)

def main(args=None):
    rclpy.init(args=args)
    aruco_service = ArucoService()
    try:
        rclpy.spin(aruco_service)
    except KeyboardInterrupt:
        pass
    finally:
        aruco_service.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
