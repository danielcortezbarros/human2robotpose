#!/usr/bin/env python3

import mediapipe as mp
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import message_filters
import json
import time 

# Decorator to measure the execution time
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result
    return wrapper


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

        with open('/root/workspace/pepper_rob_ws/src/human2robotpose/config/human2robotpose_configuration.json', 'r') as configfile:
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

    @timeit
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
            x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * self.image_width
            y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * self.image_height
            z = depth_image[int(y), int(x)]  # Use the depth map to get the Z coordinate
            left_shoulder_xyz = pixel_to_3d_camera_coords(x, y, z, self.intrinsics)

            # Right shoulder coordinates
            x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * self.image_width
            y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * self.image_height
            z = depth_image[int(y), int(x)]  # Use the depth map to get the Z coordinate
            right_shoulder_xyz = pixel_to_3d_camera_coords(x, y, z, self.intrinsics)

            # Left elbow coordinates
            x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * self.image_width
            y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * self.image_height
            z = depth_image[int(y), int(x)]
            left_elbow_xyz = pixel_to_3d_camera_coords(x, y, z, self.intrinsics)

            # Right elbow coordinates
            x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * self.image_width
            y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * self.image_height
            z= depth_image[int(y), int(x)]
            right_elbow_xyz = pixel_to_3d_camera_coords(x, y, z, self.intrinsics)

            # Left wrist coordinates
            x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * self.image_width
            y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * self.image_height
            z = depth_image[int(y), int(x)]
            left_wrist_xyz = pixel_to_3d_camera_coords(x, y, z, self.intrinsics)

            # Right wrist coordinates
            x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * self.image_width
            y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * self.image_height
            z = depth_image[int(y), int(x)]
            right_wrist_xyz = pixel_to_3d_camera_coords(x, y, z, self.intrinsics)

            # Compute 3D vectors
            LS_LE_3D = left_elbow_xyz - left_shoulder_xyz
            RS_RE_3D = right_elbow_xyz - right_shoulder_xyz
            LE_LS_3D = left_shoulder_xyz - left_elbow_xyz
            LW_LE_3D = left_elbow_xyz - left_wrist_xyz
            RE_RS_3D = right_shoulder_xyz - right_elbow_xyz
            RW_RE_3D = right_elbow_xyz - right_wrist_xyz

            LeftZeroXUpperArm = LS_LE_3D.copy()
            LeftZeroXUpperArm[0] = 0  # Zero the X component
            RightZeroXUpperArm = RS_RE_3D.copy()
            RightZeroXUpperArm[0] = 0  # Zero the X component
            LeftZeroYUpperArm = LS_LE_3D.copy()
            LeftZeroYUpperArm[1] = 0  # Zero the Y component
            RightZeroYUpperArm = RS_RE_3D.copy()
            RightZeroYUpperArm[1] = 0  # Zero the Y component

            # Lower left arm length in 2D (ignoring Z-coordinate)
            l2_left_2d = np.linalg.norm(LW_LE_3D[:2])  # X and Y components only

            # Lower right arm length in 2D (ignoring Z-coordinate)
            l2_right_2d = np.linalg.norm(RW_RE_3D[:2])  # X and Y components only

            # # Robot/Human Angles
            # LShoulderPitch = []
            # RShoulderPitch = []
            # LElbowYaw = []
            # RElbowYaw = []
            # RShoulderRoll = []
            # LShoulderRoll = []
            # RElbowRoll = []
            # LElbowRoll = []

            # Calculate the left shoulder roll angles
            tmp = (np.dot(LS_LE_3D, LeftZeroXUpperArm)) / (np.linalg.norm(LS_LE_3D) * np.linalg.norm(LeftZeroXUpperArm))
            left_shoulder_roll = np.arccos(tmp)
            if left_shoulder_roll >= 1.56:
                left_shoulder_roll = 1.56
            if left_shoulder_roll <= np.arccos((np.dot(LS_LE_3D, LeftZeroXUpperArm)) / (np.linalg.norm(LS_LE_3D) * np.linalg.norm(LeftZeroXUpperArm))):
                left_shoulder_roll = 0.0

            # Calculate the right shoulder roll angles
            tmp = (np.dot(RS_RE_3D, RightZeroXUpperArm)) / (np.linalg.norm(RS_RE_3D) * np.linalg.norm(RightZeroXUpperArm))
            right_shoulder_roll = np.arccos(tmp)
            if right_shoulder_roll >= 1.56:
                right_shoulder_roll = -1.56
            else:
                right_shoulder_roll = right_shoulder_roll * (-1)
            if right_shoulder_roll > -np.arccos((np.dot(RS_RE_3D, RightZeroXUpperArm)) / (np.linalg.norm(RS_RE_3D) * np.linalg.norm(RightZeroXUpperArm))):
                right_shoulder_roll = 0.0

            # Calculate the left elbow roll angles
            tmp = (np.dot(LE_LS_3D, LW_LE_3D)) / (np.linalg.norm(LE_LS_3D) * np.linalg.norm(LW_LE_3D))
            left_elbow_roll = np.arccos(tmp)
            if left_elbow_roll >= 1.56:
                left_elbow_roll = -1.56
            else:
                left_elbow_roll = left_elbow_roll * -1

            # Calculate the right elbow roll angles
            tmp = (np.dot(RE_RS_3D, RW_RE_3D)) / (np.linalg.norm(RE_RS_3D) * np.linalg.norm(RW_RE_3D))
            right_elbow_roll = np.arccos(tmp)
            if right_elbow_roll >= 1.56:
                right_elbow_roll = 1.56

            # Calculate the left shoulder pitch & left elbow yaw angles
            tmp = (np.dot(LeftZeroYUpperArm, LS_LE_3D)) / (np.linalg.norm(LeftZeroYUpperArm) * np.linalg.norm(LS_LE_3D))
            left_shoulder_pitch = np.arccos(tmp)
            if left_shoulder_pitch >= np.pi / 2:
                left_shoulder_pitch = np.pi / 2
            if left_shoulder_xyz[1] > left_elbow_xyz[1]:
                left_shoulder_pitch = left_shoulder_pitch * -1
            if left_shoulder_roll <= 0.4:
                left_elbow_yaw = -np.pi / 2
            elif left_elbow_xyz[1] - left_wrist_xyz[1] > 0.2 * l2_left_2d:
                left_elbow_yaw = -np.pi / 2
            elif left_elbow_xyz[1] - left_wrist_xyz[1] < 0 and -(left_elbow_xyz[1] - left_wrist_xyz[1]) > 0.2 * l2_left_2d and left_shoulder_roll > 0.7:
                left_elbow_yaw = np.pi / 2
            else:
                left_elbow_yaw = 0.0

            # Calculate the right shoulder pitch & right elbow yaw angles
            tmp = (np.dot(RightZeroYUpperArm, RS_RE_3D)) / (np.linalg.norm(RightZeroYUpperArm) * np.linalg.norm(RS_RE_3D))
            right_shoulder_pitch = np.arccos(tmp)
            if right_shoulder_pitch >= np.pi / 2:
                right_shoulder_pitch = np.pi / 2
            if right_shoulder_xyz[1] > right_elbow_xyz[1]:
                right_shoulder_pitch = right_shoulder_pitch * -1
            if right_shoulder_roll >= -0.4:
                right_elbow_yaw = np.pi / 2
            elif right_elbow_xyz[1] - right_wrist_xyz[1] > 0.2 * l2_right_2d:
                right_elbow_yaw = np.pi / 2
            elif right_elbow_xyz[1] - right_wrist_xyz[1] < 0 and -(right_elbow_xyz[1] - right_wrist_xyz[1]) > 0.2 * l2_right_2d and right_shoulder_roll < -0.7:
                right_elbow_yaw = -np.pi / 2
            else:
                right_elbow_yaw = 0.0


            # Export angles to a file for Pepper
            angles = {
                'LeftShoulderRoll': np.rad2deg(left_shoulder_roll),
                'LeftElbowRoll': np.rad2deg(left_elbow_roll),
                'LeftShoulderPitch': np.rad2deg(left_shoulder_pitch),
                'LeftElbowYaw': np.rad2deg(left_elbow_yaw),
                'RightShoulderRoll': np.rad2deg(right_shoulder_roll),
                'RightElbowRoll': np.rad2deg(right_elbow_roll),
                'RightShoulderPitch': np.rad2deg(right_shoulder_pitch),
                'RightElbowYaw': np.rad2deg(right_elbow_yaw)
            }

            print(angles)


        # Render landmarks
        self.mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Show the final image
        cv2.imshow('Mediapipe Feed', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")

