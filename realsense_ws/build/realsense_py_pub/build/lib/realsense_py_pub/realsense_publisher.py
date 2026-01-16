#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

import numpy as np
import cv2
from cv_bridge import CvBridge

import pyrealsense2 as rs


def intrinsics_to_camera_info(intr, frame_id: str) -> CameraInfo:
    """
    Convert RealSense intrinsics to ROS2 CameraInfo message.
    """
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = intr.width
    msg.height = intr.height

    # K (3x3) intrinsic camera matrix
    msg.k = [
        intr.fx, 0.0, intr.ppx,
        0.0, intr.fy, intr.ppy,
        0.0, 0.0, 1.0
    ]

    # P (3x4) projection matrix (assuming no stereo baseline)
    msg.p = [
        intr.fx, 0.0, intr.ppx, 0.0,
        0.0, intr.fy, intr.ppy, 0.0,
        0.0, 0.0, 1.0, 0.0
    ]

    # R rectification matrix (identity)
    msg.r = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ]

    # Distortion parameters
    # RealSense gives distortion model; ROS expects D array
    msg.d = list(intr.coeffs)
    msg.distortion_model = "plumb_bob"  # closest common model

    return msg


class RealSensePyPublisher(Node):
    def __init__(self):
        super().__init__("realsense_py_publisher")

        # ---------------- Params (safe defaults for Pi) ----------------
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("color_fps", 30)

        self.declare_parameter("depth_width", 640)
        self.declare_parameter("depth_height", 480)
        self.declare_parameter("depth_fps", 30)

        self.declare_parameter("enable_align_depth_to_color", True)

        self.color_width = int(self.get_parameter("color_width").value)
        self.color_height = int(self.get_parameter("color_height").value)
        self.color_fps = int(self.get_parameter("color_fps").value)

        self.depth_width = int(self.get_parameter("depth_width").value)
        self.depth_height = int(self.get_parameter("depth_height").value)
        self.depth_fps = int(self.get_parameter("depth_fps").value)

        self.enable_align = bool(self.get_parameter("enable_align_depth_to_color").value)

        # ---------------- Publishers ----------------
        self.bridge = CvBridge()

        self.pub_color = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_raw", 10)

        self.pub_color_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)
        self.pub_depth_info = self.create_publisher(CameraInfo, "/camera/depth/camera_info", 10)

        # ---------------- RealSense setup ----------------
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)
        self.config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)

        self.profile = self.pipeline.start(self.config)

        if self.enable_align:
            self.align = rs.align(rs.stream.color)
        else:
            self.align = None

        # Extract intrinsics once (for CameraInfo publishing)
        color_stream_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()

        self.color_intr = color_stream_profile.get_intrinsics()
        self.depth_intr = depth_stream_profile.get_intrinsics()

        self.color_info_msg = intrinsics_to_camera_info(self.color_intr, frame_id="camera_color_frame")
        self.depth_info_msg = intrinsics_to_camera_info(self.depth_intr, frame_id="camera_depth_frame")

        # Timer loop (publish frames)
        self.timer = self.create_timer(1.0 / max(self.color_fps, 1), self.timer_callback)

        self.get_logger().info("✅ RealSense Py Publisher started!")
        self.get_logger().info(f"Color: {self.color_width}x{self.color_height}@{self.color_fps}")
        self.get_logger().info(f"Depth: {self.depth_width}x{self.depth_height}@{self.depth_fps}")
        self.get_logger().info(f"Align depth to color: {self.enable_align}")

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)

            if self.align is not None:
                frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                self.get_logger().warn("⚠️ Missing frames (color/depth). Skipping publish.")
                return

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())  # BGR8
            depth_image = np.asanyarray(depth_frame.get_data())  # Z16 (uint16 mm)

            # Timestamp header
            now = self.get_clock().now().to_msg()
            color_header = Header(stamp=now, frame_id="camera_color_frame")
            depth_header = Header(stamp=now, frame_id="camera_depth_frame")

            # Convert to ROS Image msg
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            color_msg.header = color_header

            # depth is uint16 in mm
            depth_msg = Image()
            depth_msg.header = depth_header
            depth_msg.height = depth_image.shape[0]
            depth_msg.width = depth_image.shape[1]
            depth_msg.encoding = "16UC1"
            depth_msg.is_bigendian = False
            depth_msg.step = depth_image.shape[1] * 2
            depth_msg.data = depth_image.tobytes()

            # CameraInfo headers must match
            self.color_info_msg.header = color_header
            self.depth_info_msg.header = depth_header

            # Publish
            self.pub_color.publish(color_msg)
            self.pub_depth.publish(depth_msg)
            self.pub_color_info.publish(self.color_info_msg)
            self.pub_depth_info.publish(self.depth_info_msg)

        except Exception as e:
            self.get_logger().error(f"❌ RealSense publish error: {e}")

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSensePyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
