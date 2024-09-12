#!/usr/bin/env python3

import mediapipe as mp
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import message_filters
import json

def calculate_pepper_arm_joint_angles(shoulder_3d, elbow_3d, wrist_3d):
    # Vector from shoulder to elbow (upper arm)
    ab = np.array(elbow_3d) - np.array(shoulder_3d)
    # Vector from elbow to wrist (lower arm)
    bc = np.array(wrist_3d) - np.array(elbow_3d)
    
    # ShoulderPitch (θ1)
    shoulder_pitch = np.arctan2(elbow_3d[2] - shoulder_3d[2], elbow_3d[1] - shoulder_3d[1])
    
    # ShoulderRoll (θ2)
    shoulder_roll = np.arctan2(elbow_3d[0] - shoulder_3d[0], elbow_3d[1] - shoulder_3d[1])

    # ElbowYaw (θ3)
    elbow_yaw = np.arctan2(wrist_3d[0] - elbow_3d[0], wrist_3d[1] - elbow_3d[1])

    # ElbowRoll (θ4), angle between upper and lower arm
    dot_product = np.dot(ab, bc)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_bc = np.linalg.norm(bc)
    elbow_roll = np.arccos(dot_product / (magnitude_ab * magnitude_bc))

    # Convert angles to degrees for easy interpretation (if needed)
    shoulder_pitch = np.degrees(shoulder_pitch)
    shoulder_roll = np.degrees(shoulder_roll)
    elbow_yaw = np.degrees(elbow_yaw)
    elbow_roll = np.degrees(elbow_roll)

    return shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll
def pixel_to_3d_camera_coords(x, y, depth, K_matrix):
    """
    Convert relative 2D image coordinates (normalized between 0 and 1) and depth to 3D coordinates in world coordinates in the camera frame of reference

    Args:
        x (float): Normalized x-coordinate (between 0 and 1).
        y (float): Normalized y-coordinate (between 0 and 1).
        image_shape (tuple): Shape of the image (height, width).
        K (np.ndarray): 3x3 camera intrinsic matrix.
        depth (float): Depth value (in meters).

    Returns:
        np.ndarray: 3D position in the camera's reference frame (X, Y, Z).
    """

    # Pixel coordinates in homogeneous form [u, v, 1]
    point_2d_hom = np.array([x, y, 1])

    # Inverse of the intrinsic matrix
    K_matrix_inv = np.linalg.inv(K_matrix)

    # Apply the formula: 3D position = Z * K^-1 * pixel_coords_homogeneous
    point_3d = depth * (K_matrix_inv @ point_2d_hom)

    # Return the X, Y, Z coordinates
    return point_3d


class PoseEstimator:
    def __init__(self):

        with open('config/human2robotpose_configuration.json', 'r') as configfile:
            config = json.load(configfile)

        self.intrinsics = np.array(config['camera_intrinsics'])
        self.image_width = config['image_width']
        self.image_height = config['image_height']
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
        self.process_pose(color_image, depth_image)

    def process_pose(self, color_image, depth_image):
        # Convert to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Perform pose detection with MediaPipe
        results = self.mp_pose.process(image_rgb)

        # Recolor back to BGR for OpenCV display
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # If landmarks are detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Clip the normalized coordinates to ensure they are between 0 and 1
            left_shoulder_pixel = [np.clip(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x, 0, 1),
                                np.clip(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y, 0, 1)]
            left_elbow_pixel = [np.clip(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x, 0, 1),
                                np.clip(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y, 0, 1)]
            left_wrist_pixel = [np.clip(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x, 0, 1),
                                np.clip(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y, 0, 1)]

            # Convert normalized coordinates to pixel coordinates, ensuring values stay within the image bounds
            shoulder_pixel = [int(left_shoulder_pixel[0] * (self.image_width - 1)), int(left_shoulder_pixel[1] * (self.image_height - 1))]
            elbow_pixel = [int(left_elbow_pixel[0] * (self.image_width - 1)), int(left_elbow_pixel[1] * (self.image_height - 1))]
            wrist_pixel = [int(left_wrist_pixel[0] * (self.image_width - 1)), int(left_wrist_pixel[1] * (self.image_height - 1))]

            # Now you can safely access the depth image
            shoulder_depth = depth_image[shoulder_pixel[1], shoulder_pixel[0]]
            elbow_depth = depth_image[elbow_pixel[1], elbow_pixel[0]]
            wrist_depth = depth_image[wrist_pixel[1], wrist_pixel[0]]


            # Create 3D coordinates (x, y, z) for shoulder, elbow, and wrist
            shoulder_3d = pixel_to_3d_camera_coords(x=shoulder_pixel[0], 
                                                    y=shoulder_pixel[1], 
                                                    depth=shoulder_depth,
                                                    K_matrix=self.intrinsics)
           
            elbow_3d = pixel_to_3d_camera_coords(x=elbow_pixel[0], 
                                                    y=elbow_pixel[1], 
                                                    depth=elbow_depth,
                                                    K_matrix=self.intrinsics)
           
            wrist_3d = pixel_to_3d_camera_coords(x=wrist_pixel[0], 
                                                    y=wrist_pixel[1], 
                                                    depth=wrist_depth,
                                                    K_matrix=self.intrinsics)

            # Calculate the angle using the 3D coordinates
            angle = calculate_angle(shoulder_3d, elbow_3d, wrist_3d)

            if angle is not None:
                # Display the angle at the elbow position
                cv2.putText(image_bgr, str(int(angle)),
                            tuple(elbow_pixel), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Render landmarks
        self.mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Show the final image
        cv2.imshow('Mediapipe Feed', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")

