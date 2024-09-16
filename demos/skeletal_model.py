#!/usr/bin/env python3

import mediapipe as mp
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import message_filters

class DemoSkeletalModel:
    def __init__(self):

        self.bridge = CvBridge()  # Initialize the CvBridge class
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Subscribe to both the color and depth topics
        image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)

        # Synchronize the image and depth topics
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 0.04)
        ts.registerCallback(self.callback)

    def callback(self, image_data, depth_data):
        try:
            # Convert ROS Image message to OpenCV image for both color and depth
            color_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")  # Depth is usually 16-bit unsigned
        except CvBridgeError as e:
            rospy.logerr(f"CvBridgeError: {e}")
            return

        # Process the color image to get pose landmarks using MediaPipe
        self.show_skeletal_model(color_image, depth_image)

    def show_skeletal_model(self, color_image, depth_image):
        # Convert to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Perform pose detection with MediaPipe
        results = self.mp_pose.process(image_rgb)

        # Recolor back to BGR for OpenCV display
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Render landmarks
        self.mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Show the final image
        cv2.imshow('Mediapipe Feed', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")


def main():
    rospy.init_node('demo_skeletal_model', anonymous=True)
    demo = DemoSkeletalModel()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


