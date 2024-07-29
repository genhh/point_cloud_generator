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


class ImageSaver:
    def __init__(self):
        rospy.init_node('image_saver', anonymous=True)
        
        self.bridge = CvBridge()
        
        self.image_topics = [
            '/rt_of_low_high_res_event_cameras/optical_flow_viz',
            '/dvs_render',#8uc3
            #'/radar_depth_image',#16uc1
            '/D435i/depth/image_rect_raw',
            'rt_of_low_high_res_event_cameras/optical_flow',
            '/fliter_depth'
        ]
        
        self.images = {}
        
        for topic in self.image_topics:
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
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "8UC1" if "depth" in topic else "8UC3")
            self.images[topic] = cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def save_images(self):
        timestamp = rospy.Time.now().to_nsec()
        for topic, image in self.images.items():
            topic_name = topic.replace('/', '_')
            if topic=='rt_of_low_high_res_event_cameras/optical_flow':
                filename = os.path.join(self.save_dir, f"{topic_name}_{timestamp}.npy")
                np.save(filename,image)
                continue
            
            filename = os.path.join(self.save_dir, f"{topic_name}_{timestamp}.png")
            if "depth" in topic:
                image = cv2.applyColorMap(image,cv2.COLORMAP_JET)
                
            cv2.imwrite(filename, image)
                #cv2.imwrite(filename, image.astype(np.uint16))
            rospy.loginfo(f"Saved image: {filename}")


if __name__ == '__main__':
    rospy.init_node('image_saver', anonymous=True)
    image_saver = ImageSaver()
    rospy.spin()
