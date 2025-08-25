#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from aruco_interfaces.srv import ArucoPoseEstimate
from sensor_msgs.msg import Image, CameraInfo

class ArucoServiceClient(Node):
    def __init__(self):
        super().__init__('aruco_pose_client')
        self.cli = self.create_client(ArucoPoseEstimate, 'aruco_pose_estimate')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service /aruco_pose_estimate ...')

        self.image_msg = None
        self.camera_info_msg = None
        self.sub_img = self.create_subscription(Image, '/camera/color/image_raw', self.img_cb, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_cb, 10)

        self.timer = self.create_timer(3.0, self.send_request)

    def img_cb(self, msg):
        self.image_msg = msg

    def info_cb(self, msg):
        self.camera_info_msg = msg

    def send_request(self):
        if self.image_msg is None or self.camera_info_msg is None:
            self.get_logger().warn("No image or camera info yet, cannot send request.")
            return
        
        req = ArucoPoseEstimate.Request()
        req.image = self.image_msg
        req.camera_info = self.camera_info_msg
        future = self.cli.call_async(req)
        future.add_done_callback(self.callback)

    def callback(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return
        
        if res.success:
            pose = res.pose
            self.get_logger().info(
                f"Pose estimate success: "
                f"pos=({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}), "
                f"quat=({pose.orientation.x:.3f}, {pose.orientation.y:.3f}, "
                f"{pose.orientation.z:.3f}, {pose.orientation.w:.3f})"
            )
        else:
            self.get_logger().warn("Pose estimate failed.")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoServiceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
