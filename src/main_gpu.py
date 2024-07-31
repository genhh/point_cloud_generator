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

position_x = 260
position_y = 134

camera_intrinsics = {
            'fx': 420.836,  # Focal length in x direction
            'fy': 420.836,  # Focal length in y direction
            'cx': 424.0,  # Optical center in x direction
            'cy': 240.0   # Optical center in y direction
        }
grid_size = 2

bridge = CvBridge()

class DepthToPointCloud:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud', anonymous=True)
        rate = rospy.Rate(30)

        self.bridge = bridge
        
        self.camera_intrinsics = camera_intrinsics

        rospy.Subscriber("/Odometry_map", Odometry, self.odom_callback)
        rospy.Subscriber("/D435i/depth/image_rect_raw", Image, self.depth_image_callback)
        rospy.Subscriber('/rt_of_low_high_res_event_cameras/optical_flow', Image, self.flow_image_callback)
        
        self.point_cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=0)
        self.fliter_cloud_pub = rospy.Publisher('/fliter_cloud', PointCloud2, queue_size=0)
        #self.radar_depth_pub = rospy.Publisher('/depth_completion_image', Image, queue_size=1)
        self.fliter_depth_pub = rospy.Publisher('/fliter_depth', Image, queue_size=1)

        self.odom = None
        self.flow_image = None
        self.pre_img = None
        #self.listener = tf.TransformListener()
        #self.radar_depth_image = None

        self.depth_image_queue = Queue(maxsize=10)
        self.masked_image_queue = Queue(maxsize=10)
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
            if not self.depth_image_queue.empty() and not self.masked_image_queue.empty() and not self.odom_queue.empty():
                depth_image = self.depth_image_queue.get()
                current_odom = self.odom_queue.get()
                masked_img = self.masked_image_queue.get()

                #point_cloud = self.convert_depth_image_to_point_cloud(depth_image, current_odom)
                #self.point_cloud_pub.publish(point_cloud)
                result = self.pool.apply_async(convert_depth_image_to_point_cloud, 
                                               args=(depth_image, current_odom, False))
                result2 = self.pool.apply_async(convert_depth_image_to_point_cloud, 
                                               args=(masked_img, current_odom, True))
                self.results.append(result)
                self.results.append(result2)

            for result in self.results:
                if result.ready():
                    point_cloud, ismasked= result.get()
                    if ismasked:
                        self.fliter_cloud_pub.publish(point_cloud)
                    else:
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
            #self.pre_img = np.copy(depth_image)
            #print(depth_image.shape)
            #height, width = depth_image.shape
            #mask = generate_horizontal_grid_mask(height, width, grid_size)
            #radar_depth_image = cp.copy(depth_image)
            #radar_depth_image[mask] = 0

            #radar_depth_image_msg = bridge.cv2_to_imgmsg(cp.asnumpy(radar_depth_image), encoding="16UC1")
            #radar_depth_image_msg.header = rospy.Header()
            #self.radar_depth_pub.publish(radar_depth_image_msg)
            #print(np.max(depth_image))
            if not self.depth_image_queue.full():
                self.depth_image_queue.put(np.copy(depth_image))
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        
        if self.flow_image is not None :
            # Here you can publish or save the masked_image as needed
            
            masked_img = np.copy(depth_image)
            mask3 = masked_img > 2000#(65535//10)
            masked_img[mask3] = 0
            if not self.masked_image_queue.full():
                self.masked_image_queue.put(masked_img)
            masked_image_msg = bridge.cv2_to_imgmsg(masked_img, encoding="16UC1")
            masked_image_msg.header = rospy.Header()
            self.fliter_depth_pub.publish(masked_image_msg)
            
        
        #radar_depth_image = self.convert_depth_to_radar(depth_image)
        #point_cloud = self.convert_depth_image_to_point_cloud(depth_image, msg.header.stamp)

        #self.point_cloud_pub.publish(point_cloud)
        
        

    def odom_callback(self, msg):
        if not self.odom_queue.full():
            self.odom_queue.put(msg)

    def apply_flow_mask(self, target_img, flow_img):
        # Resize flow image if necessary
        #if flow_img.shape[:2] != target_img.shape[:2]:
        #    flow_img = cv2.resize(flow_img, (target_img.shape[1], target_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        #if flow_img.shape[:2] != target_img.shape[:2]:
        #    target_img = cv2.resize(target_img, (flow_img.shape[1], flow_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        #print(flow_img.shape)
        # Create a mask where flow values are 0; D435i minz is 0.105m
        #image_a = np.zeros((target_img.shape[0], target_img.shape[1],2), dtype=np.float32)
        #image_a[position_y:position_y+flow_img.shape[0], position_x:position_x+flow_img.shape[1],:] = flow_img
        #magnitude, angle = cv2.cartToPolar(flow_img[..., 0], flow_img[..., 1])
        #magnitude, angle = cv2.cartToPolar(image_a[..., 0], image_a[..., 1])

        # Create a mask where flow values are 0 or x and y direction ratios are not close to 1
        #mask_zero_flow = (magnitude == 0)
        #ratio_x_y = np.abs(flow_img[..., 0] / (flow_img[..., 1] + 1e-10))  # Add small value to avoid division by zero
        #mask1 = flow_img[..., 0]==0
        #mask2 = flow_img[..., 1]==0

        mask3 = target_img > 2000#(65535//10)
        target_img[mask3] = 0
        #if self.pre_img is None:
        #    self.pre_img = np.copy(target_img)
        #depth_image_masked = mask_depth_image_using_optical_flow(self.pre_img, target_img)
        #self.pre_img = np.copy(target_img)
        #mask_test = mask_zero_flow | mask3
        #mask_ratio = (ratio_x_y > 2) | (ratio_x_y < 0.5)
        
        #mask = mask_zero_flow | mask_ratio
        
        # Apply the mask to the target image
        #print(target_img.shape)
        #np.zeros(target_img)
        #target_img = detect_circles_in_image(target_img)
        #print(target_img.shape)
        #target_img = cv2.medianBlur(target_img,3)
        #print(np.max(target_img))
        return target_img
    
    """
    def publish_radar_depth_image(self, header):
        try:
            if radar_depth_image is not None:
                radar_depth_image_msg = self.bridge.cv2_to_imgmsg(cp.asnumpy(radar_depth_image), encoding="32FC1")
                radar_depth_image_msg.header = header
                self.radar_depth_pub.publish(radar_depth_image_msg)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
    """

def convert_depth_image_to_point_cloud(depth_image, odom, ismasked):
        #points = []
        scale = 2
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        # downsample, scale lager, publish hz faster
        depth_image = depth_image[::scale, ::scale]

        height, width = depth_image.shape
        u = cp.arange(width, dtype=cp.float32)
        v = cp.arange(height, dtype=cp.float32)
        u, v = cp.meshgrid(u, v)

        depth_image_gpu = cp.asarray(depth_image, dtype=cp.float32) / 1000.0  # Convert to meters

        z = depth_image_gpu
        x = (u - cx/scale) * z / fx
        y = (v - cy/scale) * z / fy

        # lidar depth
        """
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
        """     

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

        point_cloud_msg = pc2.create_cloud_xyz32(header, points)

        return point_cloud_msg, ismasked


def quaternion_to_rotation_matrix( q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

def generate_horizontal_grid_mask(height, width, grid_size):
    """
    Generates a horizontal grid mask of the specified height and width, with each grid cell having a size of grid_size.
    
    Args:
        height (int): Height of the mask.
        width (int): Width of the mask.
        grid_size (int): Size of each grid cell.
    
    Returns:
        np.ndarray: The generated mask array.
    """
    mask = np.ones((height, width), dtype=bool)
    grid_size2 = 10
    for y in range(0, height, grid_size2):
        mask[y:y + grid_size // 2, :] = False
    
    return mask

def detect_circles_in_image(img):  
  
    img = img.astype(np.float32)
    img = img.astype(np.uint8)
    # 应用双边滤波 gray scale graph  
    bilateral_filtered = cv2.bilateralFilter(img, 9, 75, 75)  
  
    # 霍夫圆检测  
    # 参数1: 灰度图  
    # 参数2: 检测方法，cv2.HOUGH_GRADIENT 是梯度法  
    # 参数3: dp = 1, 累加器分辨率与图像分辨率的反比  
    # 参数4: minDist = 100, 检测到的圆的中心之间的最小距离  
    # 参数5: param1 = 50, Canny边缘检测器的高阈值  
    # 参数6: param2 = 30, 圆心检测器累加器的阈值  
    # 参数7: minRadius = 0, 圆半径的最小值  
    # 参数8: maxRadius = 0, 圆半径的最大值  
    circles = cv2.HoughCircles(bilateral_filtered, cv2.HOUGH_GRADIENT, 1, 100,  
                               param1=50, param2=30, minRadius=0, maxRadius=0)  
  
    # 确保至少检测到一个圆  
    if circles is not None:  
        circles = np.uint16(np.around(circles))
        val = 125
        idx = 0
        idy = 0
        r = 0
        for i in circles[0, :]:
            if i[1]<img.shape[0] and i[0]<img.shape[1] and img[i[1]][i[0]] < val:
                val = img[i[1]][i[0]]
                idx = i[0]
                idy = i[1]
                r = i[2]

        # 绘制圆心  
        cv2.circle(img, (idx, idy), 1, (255, 0, 0), 3)  
        # 绘制圆轮廓  
        cv2.circle(img, (idx, idy), r, (255, 0, 255), 3) 
    
    img = img.astype(np.float32)
    img = img.astype(np.uint16)
    
    return img

def compute_dense_optical_flow(image1, image2):
    """
    计算两个图像之间的稠密光流。
    :param image1: 第一幅图像
    :param image2: 第二幅图像
    :return: 稠密光流
    """
    # 将图像转换为灰度图像
    #gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算稠密光流
    flow = cv2.calcOpticalFlowFarneback(image1, image2, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow

def create_mask_from_flow(flow):
    """
    根据稠密光流创建掩码 光流为0的位置设置为True。
    :param flow: 稠密光流
    :return: 掩码
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = (mag == 0)
    return mask

def apply_mask_to_depth_image(depth_image, mask):
    """
    对深度图像应用掩码，滤除掩码位置的深度值。
    :param depth_image: 16位无符号整型的深度图像
    :param mask: 掩码
    :return: 应用掩码后的深度图像
    """
    depth_image_masked = depth_image.copy()
    depth_image_masked[mask] = 0
    return depth_image_masked

def mask_depth_image_using_optical_flow(depth_image1, depth_image2):
    """
    使用稠密光流计算掩码，并对深度图像应用掩码。
    :param depth_image1: 第一幅16位无符号整型的深度图像
    :param depth_image2: 第二幅16位无符号整型的深度图像
    :return: 应用掩码后的深度图像
    """
    # 将16位深度图像转换为8位图像
    depth_image1_8bit = cv2.convertScaleAbs(depth_image1, alpha=(255.0/65535.0))
    depth_image2_8bit = cv2.convertScaleAbs(depth_image2, alpha=(255.0/65535.0))

    # 计算稠密光流
    flow = compute_dense_optical_flow(depth_image1_8bit, depth_image2_8bit)

    # 创建掩码
    mask = create_mask_from_flow(flow)

    # 应用掩码到深度图像
    depth_image_masked = apply_mask_to_depth_image(depth_image1, mask)

    return depth_image_masked

if __name__ == '__main__':
    try:
        depth_to_pointcloud = DepthToPointCloud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
