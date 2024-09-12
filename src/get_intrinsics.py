import pyrealsense2 as rs
import numpy as np
import json

# Create a pipeline and start the camera
pipeline = rs.pipeline()
config = rs.config()
pipeline.start(config)

# Get the depth sensor's intrinsic parameters
profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth)  # Get the depth stream
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# The intrinsic matrix K is derived from the intrinsics
fx = intrinsics.fx
fy = intrinsics.fy
cx = intrinsics.ppx  # Principal point x
cy = intrinsics.ppy  # Principal point y

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

print("Intrinsic camera matrix (K):")
print(K)

# Convert the NumPy array to a regular list so it can be serialized by JSON
K_list = K.tolist()

# Define the configuration dictionary
config_data = {
    "camera_intrinsics": K_list,
}

# Write the configuration to a JSON file
with open('config/human2robotpose_configuration.json', 'w') as configfile:
    json.dump(config_data, configfile, indent=4)

print("Configuration has been written to the JSON file.")

# Stop the pipeline
pipeline.stop()
