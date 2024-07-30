#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import os
import numpy as np
import termios
import sys
import tty
import concurrent.futures
import ros_numpy
#import pcl
from ip_basic import depth_map_utils
from sensor_msgs.msg import PointCloud2
import open3d as o3d

class ImageSaver:
    def __init__(self):
        rospy.init_node('image_saver', anonymous=True)
        
        self.bridge = CvBridge()
        
        self.image_topics = [
            '/rt_of_low_high_res_event_cameras/optical_flow_viz',
            '/dvs_render',#8uc3
            #'/radar_depth_image',#16uc1
            '/D435i/depth/image_rect_raw',
            #'rt_of_low_high_res_event_cameras/optical_flow',
            '/fliter_depth',
            '/point_cloud'
        ]
        
        self.images = {}
        
        for topic in self.image_topics:
            if topic == '/point_cloud':
                rospy.Subscriber(topic, PointCloud2, self.image_callback, topic)
                continue    
            rospy.Subscriber(topic, Image, self.image_callback, topic)
        
        self.save_dir = rospy.get_param('~save_dir', '/home/ubuntu/imgdata/')
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        rospy.Timer(rospy.Duration(0.1), self.check_key_press)
        
        


    def check_key_press(self, event):
        if self.is_key_pressed():
            self.save_images()
    
    def is_key_pressed(self):
        """Check if 's' key is pressed."""
        file_desc = sys.stdin.fileno()
        old_settings = termios.tcgetattr(file_desc)
        try:
            tty.setcbreak(file_desc)
            if sys.stdin.read(1) == 's':
                return True
        finally:
            termios.tcsetattr(file_desc, termios.TCSADRAIN, old_settings)
        return False

    def image_callback(self, msg, topic):
        try:
            if(topic=='rt_of_low_high_res_event_cameras/optical_flow'):
                cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC2")
            elif topic == '/point_cloud':
                cv_image = ros_numpy.numpify(msg)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "8UC1" if "depth" in topic else "8UC3")
            self.images[topic] = cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def save_images(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            timestamp = rospy.Time.now().to_nsec()
            
            for topic, image in self.images.items():
                topic_name = topic.replace('/', '_')
                if topic=='rt_of_low_high_res_event_cameras/optical_flow':
                    filename = os.path.join(self.save_dir, f"{topic_name}_{timestamp}.npy")
                    np.save(filename,image)
                    continue
                
                if topic=='/point_cloud':
                    filename = os.path.join(self.save_dir, f"{topic_name}_{timestamp}.pcd")
                    executor.submit(save_points,filename,image)
                    continue

                filename = os.path.join(self.save_dir, f"{topic_name}_{timestamp}.png")
                
                
                executor.submit(save_img,filename,image)    
                

def save_img(filename,image):
    if "depth" in filename:
        try:
            image = np.float32(image / 256.0)# is necessary?
            image, _ = depth_map_utils.fill_in_multiscale(image)
            
            image = (image * 256).astype(np.uint8) # only u8 can save color img
            image = cv2.applyColorMap(image,cv2.COLORMAP_JET)
        except BaseException as e:
            print("finish fill2",e)
            
    cv2.imwrite(filename, image)
    #cv2.imwrite(filename, image.astype(np.uint16))
    rospy.loginfo(f"Saved image: {filename}")

def save_points(filename,pc):
    try:
        #cloud = pcl.PointCloud_PointXYZRGB()
        #print(cloud)
        #pc = ros_numpy.numpify(pc)
        #print(pc.shape, " ::::",pc['x'].shape)
        points=np.zeros((pc.shape[0],3),dtype=np.float32)

        points[:,0]=np.copy(pc['x'])
        points[:,1]=np.copy(pc['y'])
        points[:,2]=np.copy(pc['z'])
        """
        #print(points)
        cloud.from_array(points)
        print(cloud)
        pcl.save(cloud,filename,format="pcd",binary=True)
        """
        # 创建 Open3D 点云对象
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        

        # 保存点云为 PCD 文件
        o3d.io.write_point_cloud(filename, cloud, write_ascii=False)
        rospy.loginfo(f"Saved Points: {filename}")
    except BaseException as e:
        print(e)
    


if __name__ == '__main__':
    rospy.init_node('image_saver', anonymous=True)
    image_saver = ImageSaver()
    rospy.spin()
