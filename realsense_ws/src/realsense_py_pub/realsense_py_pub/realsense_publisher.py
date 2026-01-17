#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
from std_msgs.msg import Header

import numpy as np
from cv_bridge import CvBridge

import pyrealsense2 as rs
from sensor_msgs_py import point_cloud2


def intrinsics_to_camera_info(intr: rs.intrinsics, frame_id: str) -> CameraInfo:
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = intr.width
    msg.height = intr.height

    msg.k = [
        float(intr.fx), 0.0, float(intr.ppx),
        0.0, float(intr.fy), float(intr.ppy),
        0.0, 0.0, 1.0
    ]

    msg.p = [
        float(intr.fx), 0.0, float(intr.ppx), 0.0,
        0.0, float(intr.fy), float(intr.ppy), 0.0,
        0.0, 0.0, 1.0, 0.0
    ]

    msg.r = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ]

    msg.d = [float(x) for x in intr.coeffs]
    msg.distortion_model = "plumb_bob"
    return msg


class RealSensePyPublisher(Node):
    def __init__(self):
        super().__init__("realsense_py_publisher")

        # ---------------- Params ----------------
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("color_fps", 30)

        self.declare_parameter("depth_width", 640)
        self.declare_parameter("depth_height", 480)
        self.declare_parameter("depth_fps", 30)

        self.declare_parameter("enable_align_depth_to_color", True)

        self.declare_parameter("publish_pointcloud", True)
        self.declare_parameter("pointcloud_downsample_step", 4)
        self.declare_parameter("min_valid_range_m", 0.05)

        # IMU publishing
        self.declare_parameter("publish_imu", True)
        self.declare_parameter("imu_fps", 200)

        # ---------------- Read params ----------------
        self.color_width = int(self.get_parameter("color_width").value)
        self.color_height = int(self.get_parameter("color_height").value)
        self.color_fps = int(self.get_parameter("color_fps").value)

        self.depth_width = int(self.get_parameter("depth_width").value)
        self.depth_height = int(self.get_parameter("depth_height").value)
        self.depth_fps = int(self.get_parameter("depth_fps").value)

        self.enable_align = bool(self.get_parameter("enable_align_depth_to_color").value)

        self.publish_pointcloud = bool(self.get_parameter("publish_pointcloud").value)
        self.pc_step = max(1, int(self.get_parameter("pointcloud_downsample_step").value))
        self.min_valid_range_m = float(self.get_parameter("min_valid_range_m").value)

        self.publish_imu = bool(self.get_parameter("publish_imu").value)
        self.imu_fps = int(self.get_parameter("imu_fps").value)

        self.loop_fps = max(1, min(self.color_fps, self.depth_fps))

        # ---------------- Publishers ----------------
        self.bridge = CvBridge()

        self.pub_color = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_raw", 10)

        self.pub_color_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)
        self.pub_depth_info = self.create_publisher(CameraInfo, "/camera/depth/camera_info", 10)

        self.pub_cloud = self.create_publisher(PointCloud2, "/camera/depth/points", 10)

        # ✅ RealSense IMU publisher
        self.pub_imu = self.create_publisher(Imu, "/camera/imu", 50)

        # ---------------- RealSense setup ----------------
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Video streams
        self.config.enable_stream(
            rs.stream.color,
            self.color_width,
            self.color_height,
            rs.format.bgr8,
            self.color_fps,
        )

        self.config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            self.depth_fps,
        )

        # ✅ IMU streams
        if self.publish_imu:
            self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.imu_fps)
            self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.imu_fps)

        try:
            
         self.profile = self.pipeline.start(self.config)
        
        
        except RuntimeError as e:
            self.get_logger().error(f"Pipeline start failed: {e}")
            self.get_logger().error("Disabling IMU streams and retrying...")
            self.publish_imu = False
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)
            self.config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)
            self.profile = self.pipeline.start(self.config)


        self.align = rs.align(rs.stream.color) if self.enable_align else None

        # Intrinsics for CameraInfo
        color_stream_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()

        self.color_intr = color_stream_profile.get_intrinsics()
        self.depth_intr = depth_stream_profile.get_intrinsics()

        self.color_info_msg = intrinsics_to_camera_info(self.color_intr, frame_id="camera_color_frame")
        self.depth_info_msg = intrinsics_to_camera_info(self.depth_intr, frame_id="camera_depth_frame")

        # Pointcloud calculator once
        self.pc = rs.pointcloud()

        # IMU storage (latest values)
        self.latest_accel = None  # (ax, ay, az)
        self.latest_gyro = None   # (gx, gy, gz)

        # Timer loop
        self.timer = self.create_timer(1.0 / float(self.loop_fps), self.timer_callback)

        self.get_logger().info("✅ RealSense Py Publisher started!")
        self.get_logger().info(f"Color: {self.color_width}x{self.color_height}@{self.color_fps}")
        self.get_logger().info(f"Depth: {self.depth_width}x{self.depth_height}@{self.depth_fps}")
        self.get_logger().info(f"Loop FPS: {self.loop_fps}")
        self.get_logger().info(f"Align depth to color: {self.enable_align}")
        self.get_logger().info(f"Publish PointCloud: {self.publish_pointcloud}")
        self.get_logger().info(f"Publish IMU: {self.publish_imu} @ {self.imu_fps} Hz")

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)

            # Handle alignment for image streams
            if self.align is not None:
                frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            # ✅ Extract motion frames (IMU)
            if self.publish_imu:
                for f in frames:
                    if f.is_motion_frame():
                        motion = f.as_motion_frame().get_motion_data()
                        if f.get_profile().stream_type() == rs.stream.accel:
                            self.latest_accel = (motion.x, motion.y, motion.z)
                        elif f.get_profile().stream_type() == rs.stream.gyro:
                            self.latest_gyro = (motion.x, motion.y, motion.z)

                # Publish IMU only if both are available
                if self.latest_accel is not None and self.latest_gyro is not None:
                    imu_msg = Imu()
                    imu_msg.header.stamp = self.get_clock().now().to_msg()
                    imu_msg.header.frame_id = "camera_imu_frame"

                    # RealSense gives accel in m/s^2, gyro in rad/s (good for ROS)
                    imu_msg.linear_acceleration.x = float(self.latest_accel[0])
                    imu_msg.linear_acceleration.y = float(self.latest_accel[1])
                    imu_msg.linear_acceleration.z = float(self.latest_accel[2])

                    imu_msg.angular_velocity.x = float(self.latest_gyro[0])
                    imu_msg.angular_velocity.y = float(self.latest_gyro[1])
                    imu_msg.angular_velocity.z = float(self.latest_gyro[2])

                    # Orientation is unknown (no fusion here)
                    imu_msg.orientation_covariance[0] = -1.0

                    self.pub_imu.publish(imu_msg)

            if not color_frame or not depth_frame:
                self.get_logger().warn("⚠️ Missing frames (color/depth). Skipping publish.")
                return

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            now = self.get_clock().now().to_msg()
            color_header = Header(stamp=now, frame_id="camera_color_frame")
            depth_header = Header(stamp=now, frame_id="camera_depth_frame")

            # Color msg
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            color_msg.header = color_header

            # Depth msg (16UC1)
            depth_msg = Image()
            depth_msg.header = depth_header
            depth_msg.height = depth_image.shape[0]
            depth_msg.width = depth_image.shape[1]
            depth_msg.encoding = "16UC1"
            depth_msg.is_bigendian = False
            depth_msg.step = depth_image.shape[1] * 2
            depth_msg.data = depth_image.tobytes()

            # CameraInfo headers
            self.color_info_msg.header = color_header
            self.depth_info_msg.header = depth_header

            # Publish images + info
            self.pub_color.publish(color_msg)
            self.pub_depth.publish(depth_msg)
            self.pub_color_info.publish(self.color_info_msg)
            self.pub_depth_info.publish(self.depth_info_msg)

            # ---------------- PointCloud ----------------
            if self.publish_pointcloud:
                points = self.pc.calculate(depth_frame)

                vtx_struct = np.asanyarray(points.get_vertices())
                vtx = np.stack([vtx_struct["f0"], vtx_struct["f1"], vtx_struct["f2"]], axis=-1).astype(np.float32)

                dist = np.linalg.norm(vtx, axis=1)
                valid = np.isfinite(vtx).all(axis=1) & (dist > self.min_valid_range_m)
                vtx = vtx[valid]

                if self.pc_step > 1:
                    vtx = vtx[::self.pc_step]

                cloud_msg = point_cloud2.create_cloud_xyz32(depth_header, vtx.tolist())
                self.pub_cloud.publish(cloud_msg)

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
