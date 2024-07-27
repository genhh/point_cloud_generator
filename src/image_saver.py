#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import os
import numpy as np

class ImageSaver:
    def __init__(self):
        rospy.init_node('image_saver', anonymous=True)
        
        self.bridge = CvBridge()
        
        self.image_topics = [
            '/camera/rgb/image_raw',
            '/camera/depth/image_raw'
        ]
        
        self.images = {}
        
        for topic in self.image_topics:
            rospy.Subscriber(topic, Image, self.image_callback, topic)
        
        self.save_dir = rospy.get_param('~save_dir', '/home/ubuntu/imgdata')
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.running = True
        self.run()

    def image_callback(self, msg, topic):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8" if "rgb" in topic else "16UC1")
            self.images[topic] = cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def save_images(self):
        timestamp = rospy.Time.now().to_nsec()
        for topic, image in self.images.items():
            topic_name = topic.replace('/', '_')
            filename = os.path.join(self.save_dir, f"{topic_name}_{timestamp}.png")
            if "rgb" in topic:
                cv2.imwrite(filename, image)
            else:
                cv2.imwrite(filename, image.astype(np.uint16))
            rospy.loginfo(f"Saved image: {filename}")

    def run(self):
        while not rospy.is_shutdown() and self.running:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_images()
            elif key == ord('q'):
                self.running = False

if __name__ == '__main__':
    try:
        image_saver = ImageSaver()
    except rospy.ROSInterruptException:
        pass
