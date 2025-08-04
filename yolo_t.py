import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
import time

# Load YOLO model with tracking
model = YOLO("yolo11n-pose.pt")  

# Create pipeline
pipeline = dai.Pipeline()

# Define camera sources
cam_left = pipeline.create(dai.node.MonoCamera)
cam_right = pipeline.create(dai.node.MonoCamera)
cam_rgb = pipeline.create(dai.node.ColorCamera)

# Set camera properties
cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)

# Create depth node
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)

# Link cameras to stereo depth
cam_left.out.link(stereo.left)
cam_right.out.link(stereo.right)

# Create outputs
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Connect to device
with dai.Device(pipeline) as device:
    # Output queues
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
    print("OAK-D camera started with YOLO tracking (BoTSORT). Press 'q' to quit...")
    
    # FPS calculation variables
    prev_time = 0
    curr_time = 0
    
    # Define colors for different track IDs
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
              (255, 192, 203), (144, 238, 144)]
    
    while True:
        # Get frames
        in_depth = q_depth.get()
        in_rgb = q_rgb.get()
        
        # Convert to OpenCV format
        depth_frame = in_depth.getFrame()
        rgb_frame = in_rgb.getCvFrame()
        
        # Perform object tracking using BoTSORT, only track person (class 0)
        results = model.track(rgb_frame, classes=[0], tracker="botsort.yaml", verbose=False)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Create a copy of the frame for annotation
        annotated_frame = rgb_frame.copy()
        
        # Process tracking results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get boxes, confidence scores, class IDs, and track IDs
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            # Draw tracking results
            for box, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                x1, y1, x2, y2 = box.astype(int)
                
                # Get color for this track ID
                color = colors[track_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID and confidence
                label = f"Person ID:{track_id} {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(annotated_frame, (center_x, center_y), 4, color, -1)
        
        # Display FPS and tracking info on the frame
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        
        # Display number of tracked persons
        person_count = len(track_ids) if results[0].boxes is not None and results[0].boxes.id is not None else 0
        cv2.putText(
            annotated_frame,
            f"Persons Tracked: {person_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        
        # Display frames
        cv2.imshow("OAK-D - YOLO Person Tracking (BoTSORT)", annotated_frame)
        
        # Check for quit key
        if cv2.waitKey(1) == ord('q'):
            break

# Cleanup
cv2.destroyAllWindows()
