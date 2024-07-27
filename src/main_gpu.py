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
#thread has GIL limits, not fit for compute task
from multiprocessing import Pool, Queue


odomTopic = '/Odometry_map'

camera_intrinsics = {
            'fx': 420.836,  # Focal length in x direction
            'fy': 420.836,  # Focal length in y direction
            'cx': 640.0,  # Optical center in x direction
            'cy': 360.0   # Optical center in y direction
        }

radar_depth_image = None

class DepthToPointCloud:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud', anonymous=True)
        rate = rospy.Rate(30)

        self.bridge = CvBridge()
        
        self.camera_intrinsics = camera_intrinsics

        rospy.Subscriber("/Odometry_map", Odometry, self.odom_callback)
        rospy.Subscriber("/D435i/depth/image_rect_raw", Image, self.depth_image_callback)
        rospy.Subscriber('/rt_of_low_high_res_event_cameras/optical_flow', Image, self.flow_image_callback)
        
        self.point_cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=1)
        self.radar_depth_pub = rospy.Publisher('/radar_depth_image', Image, queue_size=1)

        self.odom = None
        self.flow_image = None
        #self.listener = tf.TransformListener()
        #self.radar_depth_image = None

        self.depth_image_queue = Queue(maxsize=10)
        self.odom_queue = Queue(maxsize=10)

        #self.processing_process = Process(target=self.process_data)
        #self.processing_process.daemon = True
        #self.processing_process.start()
        cpu_count = 4
        self.pool = Pool(processes=cpu_count)  # Initialize a pool of cpu_count processes
        self.results = []

        self.process_data()
    
    def process_data(self):
        while not rospy.is_shutdown():
            if not self.depth_image_queue.empty() and not self.odom_queue.empty():
                depth_image = self.depth_image_queue.get()
                current_odom = self.odom_queue.get()

                #point_cloud = self.convert_depth_image_to_point_cloud(depth_image, current_odom)
                #self.point_cloud_pub.publish(point_cloud)
                result = self.pool.apply_async(convert_depth_image_to_point_cloud, 
                                               args=(depth_image, current_odom))
                self.results.append(result)

            for result in self.results:
                if result.ready():
                    point_cloud = result.get()
                    self.point_cloud_pub.publish(point_cloud)
                    self.results.remove(result)

    def flow_image_callback(self, msg):
        try:
            self.flow_image = self.bridge.imgmsg_to_cv2(msg, "32FC2")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_image_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            if not self.depth_image_queue.full():
                self.depth_image_queue.put(depth_image)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        
        if self.flow_image is not None:
            None
            #depth_image = self.apply_flow_mask(depth_image, self.flow_image)
            # Here you can publish or save the masked_image as needed
            # Example: self.publish_masked_image(masked_image)
        
        #radar_depth_image = self.convert_depth_to_radar(depth_image)
        #point_cloud = self.convert_depth_image_to_point_cloud(depth_image, msg.header.stamp)

        #self.point_cloud_pub.publish(point_cloud)
        self.publish_radar_depth_image(msg.header)

    def odom_callback(self, msg):
        if not self.odom_queue.full():
            self.odom_queue.put(msg)

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

    def publish_radar_depth_image(self, header):
        try:
            if radar_depth_image is not None:
                radar_depth_image_msg = self.bridge.cv2_to_imgmsg(cp.asnumpy(radar_depth_image), encoding="32FC1")
                radar_depth_image_msg.header = header
                self.radar_depth_pub.publish(radar_depth_image_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

def convert_depth_image_to_point_cloud(depth_image, odom):
        points = []

        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

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

        radar_depth_image = cp.full((height, num_distance_bins), cp.inf, dtype=cp.float32)

        angle[angle < 0] += 360
                
        angle_bin = cp.floor(angle / angle_resolution).astype(cp.int32)
        distance_bin = cp.floor(distance / distance_resolution).astype(cp.int32)

        valid_mask = (distance_bin < num_distance_bins) #and (angle_bin < height)
        radar_depth_image[angle_bin[valid_mask], distance_bin[valid_mask]] = cp.minimum(
            radar_depth_image[angle_bin[valid_mask], distance_bin[valid_mask]],
            distance[valid_mask]
        )

        radar_depth_image[cp.isinf(radar_depth_image)] = 0  # replace inf values with 0
        

        # **lidar points**
        valid = (z > 0)

        x = x[valid]
        y = y[valid]
        z = z[valid]

        points = cp.stack((x, y, z), axis=-1)
        #print(points.shape)
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

        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        matrix = cp.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        rotation_matrix_gpu = cp.asarray(rotation_matrix)
        translation_gpu = cp.asarray(translation)

        points_rotated = cp.dot(cp.dot(points, matrix.T), rotation_matrix_gpu.T )
        transformed_points = points_rotated + translation_gpu
        """
        # Simulate lidar by filtering and sampling points
        #distances = cp.linalg.norm(transformed_points, axis=1)
        angles = cp.arctan2(transformed_points[:, 1], transformed_points[:, 0])
        
        lidar_points = []
        num_beams = 70  # Number of lidar beams to simulate 70-degree range
        angle_range = np.radians(70)  # 70 degrees in radians
        angle_min = -angle_range / 2
        #angle_max = angle_range / 2
        angle_increment = angle_range / num_beams

        for i in range(num_beams):
            beam_angle_min = angle_min + i * angle_increment
            beam_angle_max = beam_angle_min + angle_increment

            mask = (angles >= beam_angle_min) & (angles < beam_angle_max)
            beam_points = transformed_points[mask]

            if beam_points.size > 0:
                min_distance_idx = cp.argmin(cp.linalg.norm(beam_points, axis=1))
                lidar_points.append(beam_points[min_distance_idx])
        
        
        points = cp.asnumpy(cp.array(lidar_points)) 
        """
        points = cp.asnumpy(transformed_points)
        
        header = rospy.Header()
        header.stamp = odom.header.stamp
        header.frame_id = odom.header.frame_id #"map"

        point_cloud = pc2.create_cloud_xyz32(header, points)

        return point_cloud


def quaternion_to_rotation_matrix( q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])


if __name__ == '__main__':
    try:
        depth_to_pointcloud = DepthToPointCloud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
