#!/usr/bin/env python

import rospy
import numpy as np
import cupy as cp
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import tf.transformations
import threading
import queue

class DepthToPointCloud:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud', anonymous=True)
        
        self.bridge = CvBridge()
        
        self.camera_intrinsics = {
            'fx': 525.0,  # Focal length in x direction
            'fy': 525.0,  # Focal length in y direction
            'cx': 319.5,  # Optical center in x direction
            'cy': 239.5   # Optical center in y direction
        }
        
        self.pointcloud_pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=1)
        
        self.depth_image_queue = queue.Queue(maxsize=10)
        self.odom_queue = queue.Queue(maxsize=10)

        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)

        self.processing_thread = threading.Thread(target=self.process_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def odom_callback(self, msg):
        if not self.odom_queue.full():
            self.odom_queue.put(msg)

    def depth_image_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            if not self.depth_image_queue.full():
                self.depth_image_queue.put((depth_image, msg.header.stamp))
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def process_data(self):
        while not rospy.is_shutdown():
            if not self.depth_image_queue.empty() and not self.odom_queue.empty():
                depth_image, timestamp = self.depth_image_queue.get()
                current_odom = self.odom_queue.get()

                point_cloud = self.convert_depth_image_to_point_cloud(depth_image, current_odom)
                self.publish_point_cloud(point_cloud, timestamp)

    def convert_depth_image_to_point_cloud(self, depth_image, odom):
        # Camera intrinsic parameters
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        height, width = depth_image.shape

        # Transfer depth image to GPU
        depth_image_gpu = cp.asarray(depth_image, dtype=cp.float32) / 1000.0  # Convert to meters

        u = cp.arange(width, dtype=cp.float32)
        v = cp.arange(height, dtype=cp.float32)
        u, v = cp.meshgrid(u, v)

        z = depth_image_gpu
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Filter valid points
        valid = (z > 0)

        x = x[valid]
        y = y[valid]
        z = z[valid]

        points = cp.stack((x, y, z), axis=-1)

        # Transform points to odom frame using GPU
        quaternion = (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w
        )

        translation = (
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z
        )

        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        rotation_matrix_gpu = cp.asarray(rotation_matrix)
        translation_gpu = cp.asarray(translation)

        points_rotated = cp.dot(points, rotation_matrix_gpu.T)
        transformed_points = points_rotated + translation_gpu

        # Simulate lidar by filtering and sampling points
        distances = cp.linalg.norm(transformed_points, axis=1)
        angles = cp.arctan2(transformed_points[:, 1], transformed_points[:, 0])

        lidar_points = []
        num_beams = 70  # Number of lidar beams to simulate 70-degree range
        angle_range = np.radians(70)  # 70 degrees in radians
        angle_min = -angle_range / 2
        angle_max = angle_range / 2
        angle_increment = angle_range / num_beams

        for i in range(num_beams):
            beam_angle_min = angle_min + i * angle_increment
            beam_angle_max = beam_angle_min + angle_increment

            mask = (angles >= beam_angle_min) & (angles < beam_angle_max)
            beam_points = transformed_points[mask]

            if beam_points.size > 0:
                min_distance_idx = cp.argmin(cp.linalg.norm(beam_points, axis=1))
                lidar_points.append(beam_points[min_distance_idx])

        return cp.asnumpy(cp.array(lidar_points))

    def quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    def publish_point_cloud(self, points, timestamp):
        header = rospy.Header()
        header.stamp = timestamp
        header.frame_id = 'odom'  # Change to your odometry frame
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(point_cloud_msg)

if __name__ == '__main__':
    try:
        depth_to_pointcloud = DepthToPointCloud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
