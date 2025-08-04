#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import argparse
import time # <--- 1. 导入 time 模块

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6

parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type=float, default=None, help="Alpha scaling parameter to increase float. [0,1] valid interval.")
args = parser.parse_args()
alpha = args.alpha

def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight


fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
# --- 修改点 1: 将单目分辨率设置为 800P，与彩色相机匹配 ---
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_800_P

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.Camera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
disparityOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
disparityOut.setStreamName("disp")
queueNames.append("disp")

#Properties
rgbCamSocket = dai.CameraBoardSocket.CAM_A

camRgb.setBoardSocket(rgbCamSocket)
camRgb.setSize(1280, 800)
camRgb.setFps(fps)

# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2()
    lensPosition = calibData.getLensPosition(rgbCamSocket)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise
left.setResolution(monoResolution)
left.setCamera("left")
left.setFps(fps)
right.setResolution(monoResolution)
right.setCamera("right")
right.setFps(fps)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# LR-check is required for depth alignment
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(rgbCamSocket)

# 明确设置输出尺寸，确保宽度是16的倍数
stereo.setOutputSize(1280, 800)


# Linking
camRgb.video.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(disparityOut.input)

# 移除重复的setSize调用
# camRgb.setSize(1280, 800)  # 这行重复了，已注释掉

# 不使用校准网格，避免自动尺寸调整
# camRgb.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
if alpha is not None:
    camRgb.setCalibrationAlpha(alpha)
    stereo.setAlphaScaling(alpha)
    # 当使用alpha校正时，强制设置RGB输出尺寸与深度匹配
    camRgb.setSize(1280, 800)

# Connect to device and start pipeline
with device:
    device.startPipeline(pipeline)

    frameRgb = None
    frameDisp = None

    # --- 2. 初始化FPS计算所需的变量 ---
    startTime = time.time()
    counter = 0
    fps = 0
    size_printed = False  # 添加标志，只打印一次尺寸信息
    # ------------------------------------

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    rgbWindowName = "rgb"
    depthWindowName = "depth"
    blendedWindowName = "rgb-depth"
    cv2.namedWindow(rgbWindowName)
    cv2.namedWindow(depthWindowName)
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)

    while True:
        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["disp"] = None

        queueEvents = device.getQueueEvents(("rgb", "disp"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]

        if latestPacket["rgb"] is not None:
            # --- 3. 计算帧率 ---
            counter+=1
            current_time = time.time()
            if (current_time - startTime) > 1:
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time
            # --------------------
            frameRgb = latestPacket["rgb"].getCvFrame()
            cv2.6(rgbWindowName, frameRgb)

        if latestPacket["disp"] is not None:
            frameDisp = latestPacket["disp"].getFrame()
            maxDisparity = stereo.initialConfig.getMaxDisparity()
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
            frameDisp = np.ascontiguousarray(frameDisp)
            cv2.imshow(depthWindowName, frameDisp)

        # Blend when both received
        if frameRgb is not None and frameDisp is not None:
            # Need to have both frames in BGR format before blending
            if len(frameDisp.shape) < 3:
                frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
            
            # 确保两个图像具有相同的尺寸
            rgb_height, rgb_width = frameRgb.shape[:2]
            disp_height, disp_width = frameDisp.shape[:2]
            
            # 只在第一次打印尺寸信息
            if not size_printed:
                print(f"RGB size: {rgb_width}x{rgb_height}, Depth size: {disp_width}x{disp_height}")
                size_printed = True
            
            # 如果尺寸不匹配，将深度图像调整为RGB图像的尺寸
            if rgb_height != disp_height or rgb_width != disp_width:
                frameDisp = cv2.resize(frameDisp, (rgb_width, rgb_height))
            
            blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)

            # --- 4. 将FPS绘制到图像上 ---
            # 为了清晰，先绘制一个黑色背景
            cv2.rectangle(blended, (0, 0), (150, 40), (0,0,0), -1)
            cv2.putText(blended, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # -----------------------------

            cv2.imshow(blendedWindowName, blended)
            frameRgb = None
            frameDisp = None

        if cv2.waitKey(1) == ord('q'):
            break