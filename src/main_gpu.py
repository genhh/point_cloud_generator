#!/usr/bin/env python3

import rospy
#import tf
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import cupy as cp

odomTopic = '/Odometry_map'


class DepthToPointCloud:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud', anonymous=True)
        
        self.bridge = CvBridge()
        
        self.camera_intrinsics = {
            'fx': 420.836,  # Focal length in x direction
            'fy': 420.836,  # Focal length in y direction
            'cx': 640.0,  # Optical center in x direction
            'cy': 360.0   # Optical center in y direction
        }

        rospy.Subscriber("/Odometry_map", Odometry, self.odom_callback)
        rospy.Subscriber("/D435i/depth/image_rect_raw", Image, self.depth_image_callback)
        rospy.Subscriber('/rt_of_low_high_res_event_cameras/optical_flow', Image, self.flow_image_callback)
        
        self.point_cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=1)
        self.radar_depth_pub = rospy.Publisher('/radar_depth_image', Image, queue_size=1)
        

        self.odom = None
        self.flow_image = None
        #self.listener = tf.TransformListener()
        self.radar_depth_image = None
    
    def flow_image_callback(self, msg):
        try:
            self.flow_image = self.bridge.imgmsg_to_cv2(msg, "32FC2")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_image_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        if self.odom is None:
            rospy.logwarn("No odom data available yet.")
            return
        
        if self.flow_image is not None:
            None
            #depth_image = self.apply_flow_mask(depth_image, self.flow_image)
            # Here you can publish or save the masked_image as needed
            # Example: self.publish_masked_image(masked_image)
        
        #radar_depth_image = self.convert_depth_to_radar(depth_image)
        point_cloud = self.convert_depth_image_to_point_cloud(depth_image, msg.header.stamp)

        self.point_cloud_pub.publish(point_cloud)
        self.publish_radar_depth_image(msg.header)

    def odom_callback(self, msg):
        self.odom = msg

    def apply_flow_mask(self, target_img, flow_img):
        # Resize flow image if necessary
        if flow_img.shape[:2] != target_img.shape[:2]:
            flow_img = cv2.resize(flow_img, (target_img.shape[1], target_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Create a mask where flow values are 0; D435i minz is 0.105m
        magnitude, angle = cv2.cartToPolar(flow_img[..., 0], flow_img[..., 1])
        
        # Create a mask where flow values are 0 or x and y direction ratios are not close to 1
        mask_zero_flow = (magnitude == 0)
        ratio_x_y = np.abs(flow_img[..., 0] / (flow_img[..., 1] + 1e-10))  # Add small value to avoid division by zero
        mask_ratio = (ratio_x_y > 1.2) | (ratio_x_y < 0.8)
        
        mask = mask_zero_flow | mask_ratio
        
        # Apply the mask to the target image
        target_img[mask] = 0#np.zeros(target_img)

        return target_img

    def convert_depth_image_to_point_cloud(self, depth_image, timestamp):
        points = []

        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        height, width = depth_image.shape
        u = cp.arange(width, dtype=cp.float32)
        v = cp.arange(height, dtype=cp.float32)
        u, v = cp.meshgrid(u, v)

        depth_image_gpu = cp.asarray(depth_image, dtype=cp.float32) / 1000.0  # Convert to meters

        z = depth_image_gpu
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # lidar depth
        angle_resolution = 360.0 / width
        distance_resolution = 0.01  # example: 1 cm per pixel

        distance = np.sqrt(x**2 + y**2 + z**2)
        angle = np.arctan2(y, x) * 180 / np.pi

        max_distance = cp.max(distance)
        num_distance_bins = int(max_distance / distance_resolution)

        self.radar_depth_image = cp.full((height, num_distance_bins), cp.inf, dtype=cp.float32)

        angle[angle < 0] += 360
                
        angle_bin = cp.floor(angle / angle_resolution).astype(cp.int32)
        distance_bin = cp.floor(distance / distance_resolution).astype(cp.int32)

        valid_mask = (distance_bin < num_distance_bins) #and (angle_bin < height)
        self.radar_depth_image[angle_bin[valid_mask], distance_bin[valid_mask]] = cp.minimum(
            self.radar_depth_image[angle_bin[valid_mask], distance_bin[valid_mask]],
            distance[valid_mask]
        )

        self.radar_depth_image[cp.isinf(self.radar_depth_image)] = 0  # replace inf values with 0
        

        # **lidar points**
        valid = (z > 0)

        x = x[valid]
        y = y[valid]
        z = z[valid]

        points = cp.stack((x, y, z), axis=-1)
        #print(points.shape)
        quaternion = (
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w
        )

        translation = (
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z
        )

        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        matrix = cp.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        rotation_matrix_gpu = cp.asarray(rotation_matrix)
        translation_gpu = cp.asarray(translation)

        points_rotated = cp.dot(cp.dot(points, matrix.T), rotation_matrix_gpu.T )
        transformed_points = points_rotated + translation_gpu
        
        header = rospy.Header()
        header.stamp = timestamp
        header.frame_id = "map"

        point_cloud = pc2.create_cloud_xyz32(header, cp.asnumpy(transformed_points))

        return point_cloud
    
    def quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])


    def publish_radar_depth_image(self, header):
        try:
            if self.radar_depth_image is not None:
                radar_depth_image_msg = self.bridge.cv2_to_imgmsg(cp.asnumpy(self.radar_depth_image), encoding="32FC1")
                radar_depth_image_msg.header = header
                self.radar_depth_pub.publish(radar_depth_image_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

            

if __name__ == '__main__':
    try:
        depth_to_pointcloud = DepthToPointCloud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
