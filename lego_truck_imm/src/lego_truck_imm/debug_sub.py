#! /usr/bin/env python3
"""
This files aims to subscribe to the IMM-filter and print its output. Used
for testing.
"""

import rospy
from lego_truck_messages.msg import PredictionIMM

def _pred_callback(msg):
    """
    Print the output
    """
    print(msg)


def run():
    """
    Run loop
    """
    rospy.init_node('lego_truck_imm_debug_sub')
    subscriber = rospy.Subscriber(
        '/predictor/predicted_trajectory_imm',
        PredictionIMM,
        _pred_callback
    )

    rate = rospy.Rate(5) # 5 Hz
    while not rospy.is_shutdown():
        rate.sleep()
