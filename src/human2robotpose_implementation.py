#!/usr/bin/env python3

import mediapipe as mp
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import message_filters
import json

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
    return point_3d[0], point_3d[1], point_3d[2]


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

        # Get pose landmarks for left shoulder, elbow, wrist
        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # Left shoulder coordinates
            X1 = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * self.image_width
            Y1 = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * self.image_height
            Z1 = depth_image[int(Y1), int(X1)]  # Use the depth map to get the Z coordinate
            X1, Y1, Z1 = pixel_to_3d_camera_coords(X1, Y1, Z1, self.intrinsics)

            # Left elbow coordinates
            X3 = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * self.image_width
            Y3 = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * self.image_height
            Z3 = depth_image[int(Y3), int(X3)]
            X3, Y3, Z3 = pixel_to_3d_camera_coords(X3, Y3, Z3, self.intrinsics)

            # Left wrist coordinates
            X5 = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * self.image_width
            Y5 = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * self.image_height
            Z5 = depth_image[int(Y5), int(X5)]
            X5, Y5, Z5 = pixel_to_3d_camera_coords(X5, Y5, Z5, self.intrinsics)

            # Compute 3D vectors
            LS_LE_3D = np.array([X3 - X1, Y3 - Y1, Z3 - Z1])  # Left Shoulder to Left Elbow
            LE_LW_3D = np.array([X5 - X3, Y5 - Y3, Z5 - Z3])  # Left Elbow to Left Wrist

            # Calculate left shoulder roll
            ZeroXLeft = LS_LE_3D.copy()
            ZeroXLeft[0] = 0  # Zero the X component

            shoulder_roll = np.arccos(np.dot(LS_LE_3D, ZeroXLeft) / (np.linalg.norm(LS_LE_3D) * np.linalg.norm(ZeroXLeft)))
            shoulder_roll = min(shoulder_roll, np.pi / 2)  # Limit to max range of 90 degrees

            # Calculate left elbow roll
            elbow_roll = np.arccos(np.dot(LS_LE_3D, LE_LW_3D) / (np.linalg.norm(LS_LE_3D) * np.linalg.norm(LE_LW_3D)))
            elbow_roll = -min(elbow_roll, np.pi / 2)  # Limit to max range of 90 degrees, negative for Pepper's convention

            # Calculate left shoulder pitch
            ZeroYLeft = LS_LE_3D.copy()
            ZeroYLeft[1] = 0  # Zero the Y component
            shoulder_pitch = np.arccos(np.dot(LS_LE_3D, ZeroYLeft) / (np.linalg.norm(LS_LE_3D) * np.linalg.norm(ZeroYLeft)))

            # Calculate left elbow yaw
            if shoulder_roll <= 0.4:
                elbow_yaw = -np.pi / 2
            elif Y3 - Y5 > 0.2 * np.linalg.norm(LE_LW_3D):
                elbow_yaw = -np.pi / 2
            else:
                elbow_yaw = 0.0

            # Export angles to a file for Pepper
            angles = {
                'LShoulderRoll': [np.rad2deg(shoulder_roll)],
                'LElbowRoll': [np.rad2deg(elbow_roll)],
                'LShoulderPitch': [np.rad2deg(shoulder_pitch)],
                'LElbowYaw': [np.rad2deg(elbow_yaw)]
            }

            print(angles)


        # Render landmarks
        self.mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Show the final image
        cv2.imshow('Mediapipe Feed', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")

