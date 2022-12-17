#! /usr/bin/env python3
"""
This files aims to publish to the IMM-filter. Used
for testing.
"""

import rospy
import numpy as np
from lego_truck_messages.msg import DynamicObstacleIMM

def next_state(xk):
    """
    Motion model for pedestrian in
    Constant Velocity mode.
    Returns the updated states.
    """
    Ts = 0.5
    return np.array([
        xk[0] + Ts*xk[2]*np.cos(xk[3]),    #x-pos
        xk[1] + Ts*xk[2]*np.sin(xk[3]),    #y-pos
        xk[2],                             #velocity
        xk[3],                             #heading
        0                                  #angular velocity
    ])


def run():
    """
    Run loop
    """
    rospy.init_node('lego_truck_imm_debug_pub')
    publisher = rospy.Publisher('/obstacle_simulator/dynamic_obstacle_imm',
                                DynamicObstacleIMM, queue_size=5)
    rate = rospy.Rate(5) # 5 Hz
    xk = np.array([1,1,1,np.pi/4,np.deg2rad(0.01)])

    while not rospy.is_shutdown():
        msg = DynamicObstacleIMM()
        msg.header.stamp = rospy.get_rostime()
        msg.x = xk[0] #+ np.random.normal(0, 0.1)
        msg.y = xk[1] #+ np.random.normal(0, 0.1)
        msg.theta = xk[3] + np.random.normal(0, 0.01)
        msg.type = 0 #Pedestrian
        xk = next_state(xk)
        publisher.publish(msg)
        rospy.loginfo("DEBUG PUB: Publishing data")
        rate.sleep()
