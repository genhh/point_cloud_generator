#!/usr/bin/env python

import rospy
import tf
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

odomTopic = '/Odometry_map'

class DepthToPointCloud:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud', anonymous=True)
        
        self.bridge = CvBridge()
        
        self.camera_intrinsics = {
            'fx': 910.0,  # Focal length in x direction
            'fy': 907.0,  # Focal length in y direction
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
        self.listener = tf.TransformListener()
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
        # Precompute angle and distance resolution
        angle_resolution = 360.0 / width
        distance_resolution = 0.01  # example: 1 cm per pixel

        max_distance = np.max(depth_image) / 1000.0  # convert from mm to meters
        num_distance_bins = int(max_distance / distance_resolution)

        self.radar_depth_image = np.full((height, num_distance_bins), np.inf, dtype=np.float32)

        for v in range(height):
            for u in range(width):
                z = depth_image[v, u] / 1000.0
                if z == 0:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                point = [x, y, z]
                points.append(point)

                distance = np.sqrt(x**2 + y**2 + z**2)
                angle = np.arctan2(y, x) * 180 / np.pi

                if angle < 0:
                    angle += 360.0
                
                angle_bin = int(angle / angle_resolution)
                distance_bin = int(distance / distance_resolution)

                if distance_bin < num_distance_bins and angle_bin < height:
                    self.radar_depth_image[angle_bin, distance_bin] = min(self.radar_depth_image[angle_bin, distance_bin], distance)
        
        self.radar_depth_image[np.isinf(self.radar_depth_image)] = 0  # replace inf values with 0

        header = rospy.Header()
        header.stamp = timestamp
        header.frame_id = self.odom.child_frame_id

        point_cloud = pc2.create_cloud_xyz32(header, points)

        return self.transform_point_cloud(point_cloud)

    def transform_point_cloud(self, point_cloud):
        if self.odom is None:
            return point_cloud

        transformed_points = []
        position = self.odom.pose.pose.position
        orientation = self.odom.pose.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        translation = (position.x, position.y, position.z)

        for point in pc2.read_points(point_cloud, field_names=("x", "y", "z"), skip_nans=True):
            transformed_point = self.apply_transform(point, translation, quaternion)
            transformed_points.append(transformed_point)

        header = rospy.Header()
        header.stamp = point_cloud.header.stamp
        header.frame_id = "map"

        transformed_point_cloud = pc2.create_cloud_xyz32(header, transformed_points)

        return transformed_point_cloud
    
    def apply_transform(self, point, translation, quaternion):
        point = np.array(point)
        matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) 
        rot_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
        translated_point = np.dot(rot_matrix, np.dot(matrix, point)) + translation

        return translated_point

    def publish_radar_depth_image(self, header):
        try:
            if self.radar_depth_image is not None:
                radar_depth_image_msg = self.bridge.cv2_to_imgmsg(self.radar_depth_image, encoding="32FC1")
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
