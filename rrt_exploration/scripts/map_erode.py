#!/usr/bin/env python3
# coding: utf-8
import cv2
import rospy
import numpy as np
from copy import copy
from nav_msgs.msg import OccupancyGrid
from key_params import params

map_topic = rospy.get_param('~map_topic', '/projected_map_erode')
kernel_size = params['scene1']['erode_kernal_size']
kernel_dilate = np.ones((2, 2), np.uint8)
kernel = np.ones((kernel_size, kernel_size), np.uint8)
pub = rospy.Publisher(map_topic, OccupancyGrid, queue_size=10)
new_msg = OccupancyGrid()

OBSTACLE = 100

def map_callback(msg):
    global new_msg, pub, kernel
    map_array = np.asarray(msg.data).reshape((msg.info.height,
                                        msg.info.width))
    binary = np.zeros_like(map_array, np.uint8)
    binary.fill(255)  # white
    binary[np.where(map_array == OBSTACLE)] = 0  # black
    
    erode = cv2.erode(binary, kernel)

    
    map_array[np.where(erode == 0)] = OBSTACLE # 0 is black
    new_msg.info = msg.info
    new_msg.header = msg.header
    new_msg.data = tuple(map_array.ravel())
    assert len(new_msg.data) == len(msg.data), f'{len(new_msg.data)}, {len(msg.data)}'
    pub.publish(copy(new_msg))
    

if __name__ == '__main__':
    rospy.init_node('map_erode', anonymous=False)
    rateHz = rospy.get_param('~rateHz', 100)
    rate = rospy.Rate(rateHz)
    raw_map_topic = rospy.get_param('~raw_map_topic', '/projected_map')
    rospy.Subscriber(raw_map_topic, OccupancyGrid, map_callback)
    while not rospy.is_shutdown():
        rate.sleep()
    