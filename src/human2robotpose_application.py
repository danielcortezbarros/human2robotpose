#!/usr/bin/env python3

import rospy
import cv2
from human2robotpose_implementation import PoseEstimator  # Import the class and function

def main():
    rospy.init_node('human_pose_estimator', anonymous=True)
    pose_estimator = PoseEstimator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
