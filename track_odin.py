# 文件名: track_torch_ros.py
# 描述: 通过订阅ROS2话题获取图像，自动选择中心目标进行跟踪。
# 版本: v5.0 - ROS2集成版
#
# 运行依赖:
# - rclpy
# - cv_bridge
# - sensor_msgs
# - message_filters

# ros2 topic pub /start_vision std_msgs/msg/Int32 "{data: 1}"
import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import sys
import os
import threading
import queue
import grpc
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import json

# ==============================================================================
# ROS2 相关导入
# ==============================================================================
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError
import message_filters

# 导入生成的gRPC模块
try:
    import tracking_pb2
    import tracking_pb2_grpc
except ImportError:
    print("警告: 未找到gRPC模块，gRPC通信功能将被禁用")
    print("请运行: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. tracking.proto")
    tracking_pb2 = None
    tracking_pb2_grpc = None

# ReID 相关导入
from reid.data.transforms import build_transforms
from reid.config import cfg as reidCfg
from reid.modeling import build_model
from utils.plotting import plot_one_box

# 导入扩展卡尔曼滤波器
from extended_kalman_filter import ExtendedKalmanFilter3D, AdaptiveEKF3D, EnhancedEKF3D


# ==============================================================================
# 姿态可视化相关
# ==============================================================================

# COCO 格式的人体骨架连接 (17个关键点)
SKELETON_CONNECTIONS = [
    # 头部
    (0, 1), (0, 2), (1, 3), (2, 4),  # 鼻子-眼睛-耳朵
    # 身体中轴线
    (5, 6), (5, 11), (6, 12), (11, 12),  # 肩膀-髋部
    # 左臂
    (5, 7), (7, 9),  # 左肩-左肘-左腕
    # 右臂
    (6, 8), (8, 10),  # 右肩-右肘-右腕
    # 左腿
    (11, 13), (13, 15),  # 左髋-左膝-左踝
    # 右腿
    (12, 14), (14, 16)   # 右髋-右膝-右踝
]

# 关键点颜色 (BGR格式)
KEYPOINT_COLORS = [
    (255, 0, 0),    # 0: 鼻子 - 红色
    (255, 85, 0),   # 1: 左眼 - 橙红色
    (255, 170, 0),  # 2: 右眼 - 橙色
    (255, 255, 0),  # 3: 左耳 - 黄色
    (170, 255, 0),  # 4: 右耳 - 黄绿色
    (85, 255, 0),   # 5: 左肩 - 绿色
    (0, 255, 0),    # 6: 右肩 - 纯绿色
    (0, 255, 85),   # 7: 左肘 - 青绿色
    (0, 255, 170),  # 8: 右肘 - 青色
    (0, 255, 255),  # 9: 左腕 - 青蓝色
    (0, 170, 255),  # 10: 右腕 - 蓝色
    (0, 85, 255),   # 11: 左髋 - 蓝紫色
    (0, 0, 255),    # 12: 右髋 - 紫色
    (85, 0, 255),   # 13: 左膝 - 紫红色
    (170, 0, 255),  # 14: 右膝 - 粉色
    (255, 0, 255),  # 15: 左踝 - 品红色
    (255, 0, 170)   # 16: 右踝 - 玫红色
]

def draw_keypoints(image, keypoints, keypoints_conf, conf_threshold=0.1):
    """绘制关键点"""
    for i, (kpt, conf) in enumerate(zip(keypoints, keypoints_conf)):
        if conf > conf_threshold:
            x, y = int(kpt[0]), int(kpt[1])
            color = KEYPOINT_COLORS[i] if i < len(KEYPOINT_COLORS) else (255, 255, 255)
            cv2.circle(image, (x, y), 4, color, -1)
            cv2.circle(image, (x, y), 6, (0, 0, 0), 2)  # 黑色边框

def draw_skeleton(image, keypoints, keypoints_conf, conf_threshold=0.1):
    """绘制骨架连接"""
    for connection in SKELETON_CONNECTIONS:
        pt1_idx, pt2_idx = connection
        if (pt1_idx < len(keypoints_conf) and pt2_idx < len(keypoints_conf) and
            keypoints_conf[pt1_idx] > conf_threshold and keypoints_conf[pt2_idx] > conf_threshold):
            
            pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
            pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
            
            # 根据连接部位选择颜色
            if connection in [(5, 6), (5, 11), (6, 12), (11, 12)]:  # 躯干
                color = (0, 255, 0)  # 绿色
            elif connection in [(5, 7), (7, 9)]:  # 左臂
                color = (255, 0, 0)  # 蓝色
            elif connection in [(6, 8), (8, 10)]:  # 右臂
                color = (0, 0, 255)  # 红色
            elif connection in [(11, 13), (13, 15)]:  # 左腿
                color = (255, 255, 0)  # 青色
            elif connection in [(12, 14), (14, 16)]:  # 右腿
                color = (0, 255, 255)  # 黄色
            else:  # 头部
                color = (255, 0, 255)  # 品红色
            
            cv2.line(image, pt1, pt2, color, 2)

def draw_pose_on_person(image, detection, label_prefix="", label_color=(0, 255, 0)):
    """在检测到的人物上绘制姿态骨架和边界框"""
    # 绘制边界框
    plot_one_box(detection['box'], image, label=label_prefix, color=label_color)
    
    # 绘制骨架和关键点
    draw_skeleton(image, detection['keypoints'], detection['keypoints_conf'])
    draw_keypoints(image, detection['keypoints'], detection['keypoints_conf'])
    
    # 绘制人体中心点
    body_center = calculate_body_center_from_keypoints(
        detection['keypoints'], detection['keypoints_conf'], detection['box']
    )
    center_x, center_y = int(body_center[0]), int(body_center[1])
    cv2.circle(image, (center_x, center_y), 8, (0, 255, 255), -1)  # 黄色圆点
    cv2.circle(image, (center_x, center_y), 10, (0, 0, 0), 2)     # 黑色边框

# ==============================================================================
# 坐标导出器 (用于ROS2集成)
# ==============================================================================
class CoordinateExporter:
    def __init__(self, export_file='/tmp/tracking_coords.json'):
        self.export_file = export_file
        self.last_coords = None
        
    def export_coordinates(self, coords_tuple):
        """导出坐标到文件，供ROS2节点读取"""
        try:
            if coords_tuple:
                x, y, z = coords_tuple
            else:
                x, y, z = 0.0, 0.0, 0.0
            
            # 只有坐标发生变化时才写入文件
            current_coords = (x, y, z)
            if current_coords == self.last_coords:
                return
                
            data = {
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'timestamp': time.time()
            }
            
            with open(self.export_file, 'w') as f:
                json.dump(data, f)
            
            self.last_coords = current_coords
            
        except Exception as e:
            print(f"❌ 导出坐标时出错: {e}")


# ==============================================================================
# gRPC 客户端
# ==============================================================================
class TrackingGRPCClient:
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.connected = False
        self.coordinate_queue = queue.Queue(maxsize=100)
        self.stream_thread = None
        self.streaming = False

    def connect(self):
        if not all([tracking_pb2, tracking_pb2_grpc]):
            print("gRPC模块未导入，跳过连接")
            return False
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            grpc.channel_ready_future(self.channel).result(timeout=5)
            self.stub = tracking_pb2_grpc.TrackingServiceStub(self.channel)
            self.connected = True
            print(f"✅ gRPC客户端连接成功: {self.server_address}")
            self.start_coordinate_stream()
            return True
        except Exception as e:
            print(f"❌ gRPC连接异常: {e}")
            self.connected = False
            return False

    def start_coordinate_stream(self):
        if not self.connected: return

        def coordinate_generator():
            while self.streaming:
                try:
                    coordinate = self.coordinate_queue.get(timeout=5.0)
                    yield coordinate
                except queue.Empty:
                    continue

        def stream_worker():
            try:
                self.streaming = True
                print("📡 坐标流传输已启动...")
                self.stub.SendCoordinates(coordinate_generator())
            except grpc.RpcError as e:
                if e.code() != grpc.StatusCode.CANCELLED:
                    print(f"❌ 坐标流传输RPC失败: {e.code()} - {e.details()}")
            finally:
                self.streaming = False
                print("📡 坐标流传输已停止。")

        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()

    def disconnect(self):
        self.streaming = False
        if self.channel:
            self.channel.close()
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2)
        self.connected = False
        print("gRPC客户端已断开连接")

    def send_target_coordinates(self, coords_tuple):
        if not self.connected or not self.streaming: return

        try:
            if coords_tuple:
                x, y, z = coords_tuple
            else:
                x, y, z = 0.0, 0.0, 0.0
            
            coordinate_msg = tracking_pb2.CoordinateData(x=float(x), y=float(y), z=float(z))
            
            if self.coordinate_queue.full():
                self.coordinate_queue.get_nowait()
            self.coordinate_queue.put_nowait(coordinate_msg)
        except Exception as e:
            print(f"❌ 发送坐标到队列时出错: {e}")

    def get_command_state(self):
        if not self.connected:
            return False, 0
        try:
            status = self.stub.GetTrackingStatus(tracking_pb2.Empty(), timeout=0.5)
            return status.is_active, status.target_id
        except grpc.RpcError:
            return False, 0


# ==============================================================================
# 核心逻辑
# ==============================================================================
def calculate_3d_coordinates(depth_map, center_point, size=None):
    u, v = int(center_point[0]), int(center_point[1])
    height, width = depth_map.shape
    w, h = (10, 10) if size is None else size
    roi_size = max(5, int(min(w, h) * 0.1))
    x1, y1 = max(0, u - roi_size), max(0, v - roi_size)
    x2, y2 = min(width - 1, u + roi_size), min(height - 1, v + roi_size)
    if x1 >= x2 or y1 >= y2: return (0, 0, 0)
    
    depth_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
    
    # 深度值单位已经是米 (32FC1)，所以不需要除以1000
    # 过滤掉无效的深度值 (0 或 NaN)
    valid_mask = (depth_roi > 0.3) & (depth_roi < 15.0) & ~np.isnan(depth_roi)
    
    if not np.any(valid_mask): return (0, 0, 0)
    
    median_depth = np.median(depth_roi[valid_mask])
    Z_cam = median_depth
    
    if Z_cam <= 0.3 or Z_cam > 15.0: return (0, 0, 0)
    
    # 使用Odin相机的内参
    fx, fy = 734.357, 734.629
    cx, cy = 816.469, 642.979
    
    try:
        X_cam = (u - cx) * Z_cam / fx
        Y_cam = (v - cy) * Z_cam / fy
        # 坐标系转换：(相机坐标系 -> 世界/机器人坐标系)
        # X_world -> 前方, Y_world -> 左方, Z_world -> 上方
        X_world = Z_cam
        Y_world = -X_cam
        Z_world = -Y_cam
    except ZeroDivisionError: return (0, 0, 0)
    
    if any(math.isnan(val) for val in (X_world, Y_world, Z_world)): return (0, 0, 0)
    
    return (X_world, Y_world, Z_world)

def detect_all_poses(frame, model, conf_thres=0.5):
    """检测所有人的姿态，返回包含边界框和关键点的检测结果"""
    results = model.predict(source=frame, show=False, classes=[0], conf=conf_thres, verbose=False)
    detections = []
    if len(results[0].boxes) > 0 and results[0].keypoints is not None:
        for i in range(len(results[0].boxes)):
            box = results[0].boxes[i]
            if box.conf[0] > conf_thres:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
                if (xmax - xmin) * (ymax - ymin) > 2000:
                    keypoints = results[0].keypoints[i].xy.cpu().numpy()[0]
                    keypoints_conf = results[0].keypoints[i].conf.cpu().numpy()[0]
                    detections.append({
                        'box': (xmin, ymin, xmax, ymax),
                        'keypoints': keypoints,
                        'keypoints_conf': keypoints_conf
                    })
    return detections

def calculate_body_center_from_keypoints(keypoints, keypoints_conf, bbox):
    """
    使用四个关键点计算人体中心：左右肩膀(5,6)和左右髋部(11,12)
    如果关键点不可用，则回退到边界框中心
    """
    # COCO格式关键点索引
    left_shoulder, right_shoulder = 5, 6
    left_hip, right_hip = 11, 12
    
    # 收集有效的关键点
    valid_points = []
    conf_threshold = 0.01
    
    if keypoints_conf[left_shoulder] > conf_threshold:
        valid_points.append(keypoints[left_shoulder])
    if keypoints_conf[right_shoulder] > conf_threshold:
        valid_points.append(keypoints[right_shoulder])
    if keypoints_conf[left_hip] > conf_threshold:
        valid_points.append(keypoints[left_hip])
    if keypoints_conf[right_hip] > conf_threshold:
        valid_points.append(keypoints[right_hip])
    
    # 如果有足够的关键点，计算中心
    if len(valid_points) >= 2:
        valid_points = np.array(valid_points)
        center_x = np.mean(valid_points[:, 0])
        center_y = np.mean(valid_points[:, 1])
        return (center_x, center_y)
    else:
        # 回退到边界框中心
        xmin, ymin, xmax, ymax = bbox
        return ((xmin + xmax) / 2, (ymin + ymax) / 2)

def detect_all_persons(frame, model, conf_thres=0.5):
    """兼容函数：从pose检测中提取边界框"""
    pose_detections = detect_all_poses(frame, model, conf_thres)
    boxes = []
    for det in pose_detections:
        boxes.append(det['box'])
    return boxes

def find_center_person(frame, yolo_model):
    """在所有检测到的人中，找到最接近画面中心的一个，返回姿态检测结果"""
    detections = detect_all_poses(frame, yolo_model)
    if not detections: return None
    
    frame_center_x, frame_center_y = frame.shape[1] / 2, frame.shape[0] / 2
    min_dist = float('inf')
    center_detection = None
    
    for det in detections:
        # 使用关键点计算人体中心
        body_center = calculate_body_center_from_keypoints(
            det['keypoints'], det['keypoints_conf'], det['box']
        )
        
        # 计算到画面中心的距离
        dist = math.sqrt((body_center[0] - frame_center_x)**2 + 
                        (body_center[1] - frame_center_y)**2)
        if dist < min_dist:
            min_dist = dist
            center_detection = det
    
    return center_detection


# ==============================================================================
# 多线程框架
# ==============================================================================
class ProcessingThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, stop_event, start_event, grpc_client, args, yolo_model, reid_model):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.start_event = start_event
        self.grpc_client = grpc_client
        self.args = args
        self.device = torch.device(args.device)
        self.yolo_model = yolo_model.to(self.device)
        self.reid_model = reid_model.to(self.device)
        
        # 坐标导出器 (用于ROS2集成)
        self.coord_exporter = CoordinateExporter() if not args.no_ros_export else None
        
        # ROS2 控制文件读取 (不影响gRPC逻辑)
        self.ros_control_file = '/tmp/vision_control.json'
        self.last_ros_check_time = 0
        self.ros_command_active = False
        self.last_ros_command_active = False  # 记录上一次的ROS2命令状态
        
        # 状态机相关
        self.state = 'IDLE'
        self.query_feats = None
        self.captured_features = []
        self.capture_start_time = 0
        self.last_capture_time = 0
        self.last_grpc_check_time = 0
        self.current_depth_frame = None
        
        # 扩展卡尔曼滤波器初始化 - 使用增强版EKF
        print(f"🎯 使用增强版卡尔曼滤波器 (包含角速度的匀加速运动模型)")
        self.ekf = EnhancedEKF3D(
            process_noise_std=getattr(args, 'ekf_process_noise', 2.0),
            measurement_noise_std=getattr(args, 'ekf_measurement_noise', 8.0),
            initial_velocity_std=getattr(args, 'ekf_velocity_std', 0.5),
            initial_acceleration_std=getattr(args, 'ekf_acceleration_std', 0.3),
            initial_angular_velocity_std=getattr(args, 'ekf_angular_velocity_std', 0.2)
        )
        
        # 可视化相关
        self.last_tracked_bbox = None
        self.last_tracked_kpts = None
        self.last_tracked_kpts_conf = None
        self.last_match_dist = 0.0
        self.last_coords = None
        self.last_filtered_coords = None  # 滤波后的坐标
        self.last_predicted_coords = None  # 预测的坐标
        self.status_message = "状态: 待机 (等待指令...)"
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.enable_visualization = not args.no_viz

    def run(self):
        if self.grpc_client: self.grpc_client.connect()
        build_transforms(reidCfg)

        while not self.stop_event.is_set():
            try:
                frame, depth_frame = self.frame_queue.get(timeout=1)
                self.current_depth_frame = depth_frame
            except queue.Empty:
                continue
            
            # 1. 执行核心状态逻辑
            self.handle_state(frame)

            # 2. 如果启用，创建并发送可视化帧
            if self.enable_visualization:
                vis_frame = self.create_visualization(frame)
                if self.result_queue.full():
                    self.result_queue.get_nowait()
                self.result_queue.put(vis_frame)

        if self.grpc_client: self.grpc_client.disconnect()
        print("处理线程已停止。")
        
    def handle_state(self, frame):
        start_signal = self.check_start_signal()

        if self.state == 'IDLE':
            self.status_message = "状态: 待机 (按R或等待gRPC/ROS2指令)"
            if start_signal:
                self.transition_to_capturing(frame)
        
        elif self.state == 'CAPTURING':
            self.process_capturing(frame)
        elif self.state == 'TRACKING':
            self.process_tracking(frame)
            # 检查gRPC停止信号 (保持原有逻辑不变)
            if self.grpc_client and (time.time() - self.last_grpc_check_time > 0.2):
                self.last_grpc_check_time = time.time()
                is_active, _ = self.grpc_client.get_command_state()
                if not is_active and self.grpc_client.connected and not self.last_ros_command_active:
                    print("收到gRPC停止指令，返回待机状态。")
                    self.transition_to_idle()
                    return
            
            # 检查ROS2停止信号 (新增，不影响gRPC)
            # 如果之前是通过ROS2启动的，检查ROS2信号是否从1变为0
            if hasattr(self, '_started_by_ros') and self._started_by_ros:
                prev_ros_active = self.last_ros_command_active
                current_ros_active = self.check_ros_control_signal()
                # 如果ROS2信号从激活变为非激活，则停止跟踪
                if prev_ros_active and not current_ros_active:
                    print("检测到ROS2信号从1变为0，返回待机状态。")
                    self.transition_to_idle()
            else:
                # 即使不是ROS2启动的，也要更新ROS2状态
                self.check_ros_control_signal()

    def check_ros_control_signal(self):
        """检查ROS2控制信号 (不影响gRPC逻辑)"""
        if time.time() - self.last_ros_check_time < 0.5:  # 每0.5秒检查一次
            return self.ros_command_active
            
        self.last_ros_check_time = time.time()
        
        try:
            if not os.path.exists(self.ros_control_file):
                self.ros_command_active = False
                self.last_ros_command_active = False
                return False
                
            with open(self.ros_control_file, 'r') as f:
                data = json.load(f)
                command = data.get('command', 0)
                
                new_active = (command == 1)
                
                # 状态改变时打印日志
                if new_active != self.last_ros_command_active:
                    if new_active:
                        print("收到ROS2开启跟随指令...")
                    else:
                        print("收到ROS2关闭跟随指令...")
                
                self.last_ros_command_active = new_active
                self.ros_command_active = new_active
                return self.ros_command_active
                
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            self.ros_command_active = False
            self.last_ros_command_active = False
            return False

    def check_start_signal(self):
        # 1. 检查键盘信号
        if self.start_event.is_set():
            self.start_event.clear()
            self._started_by_ros = False
            print("收到 'R' 键信号，准备开始捕获...")
            return True
            
        # 2. 检查gRPC信号 (保持原有逻辑不变)
        if self.grpc_client and (time.time() - self.last_grpc_check_time > 0.2):
            self.last_grpc_check_time = time.time()
            is_active, _ = self.grpc_client.get_command_state()
            if is_active:
                self._started_by_ros = False
                print("收到gRPC开始指令，准备开始捕获...")
                return True
                
        # 3. 检查ROS2控制信号 (新增，不影响gRPC)
        prev_ros_active = self.last_ros_command_active
        current_ros_active = self.check_ros_control_signal()
        
        # 只有在从非激活状态变为激活状态时才触发开始信号
        if current_ros_active and not prev_ros_active and self.state == 'IDLE':
            self._started_by_ros = True
            print("检测到ROS2信号从0变为1，准备开始捕获...")
            return True
            
        return False

    def transition_to_capturing(self, frame):
        initial_detection = find_center_person(frame, self.yolo_model)
        if initial_detection is None:
            print("启动失败：画面中央未检测到目标。")
            return
        self.state = 'CAPTURING'
        self.captured_features = []
        self.capture_start_time = time.time()
        self.last_capture_time = time.time() - 1.9
        self.status_message = "collecting... (0/5)"
        print(f"目标锁定：{initial_detection['box']}。开始特征捕获...")

    def process_capturing(self, frame):
        time_elapsed = time.time() - self.capture_start_time
        if time_elapsed > 3.0:
            if len(self.captured_features) > 0:
                print(f"特征捕获完成，共 {len(self.captured_features)} 个。正在融合特征...")
                feats_tensor = torch.cat(self.captured_features, dim=0)
                avg_feat = torch.mean(feats_tensor, dim=0, keepdim=True)
                self.query_feats = F.normalize(avg_feat, dim=1, p=2)
                self.transition_to_tracking()
            else:
                print("捕获失败，未采集到任何有效特征。")
                self.transition_to_idle()
            return
        if len(self.captured_features) < 5 and (time.time() - self.last_capture_time) > 0.6:
            detection = find_center_person(frame, self.yolo_model)
            if detection:
                (xmin, ymin, xmax, ymax) = detection['box']
                crop_img = frame[ymin:ymax, xmin:xmax]
                if crop_img.size > 0:
                    crop_img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    img_tensor = build_transforms(reidCfg)(crop_img_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.reid_model(img_tensor)
                    self.captured_features.append(feat)
                    self.last_capture_time = time.time()
                    self.status_message = f"collecting... ({len(self.captured_features)}/5)"
                    print(f"已捕获特征 {len(self.captured_features)}/5")

    def process_tracking(self, frame):
        if self.query_feats is None:
            self.transition_to_idle()
            return
        self.status_message = "tracking..."
        person_detections = detect_all_poses(frame, self.yolo_model, self.args.conf_thres)
        best_match_info = None
        current_time = time.time()
        
        if person_detections:
            gallery_locs, gallery_feats = self.extract_gallery_features(frame, person_detections)
            if gallery_feats is not None:
                distmat = self.calculate_distance_matrix(gallery_feats)
                best_g_idx = np.argmin(distmat[0])
                min_dist = distmat[0, best_g_idx]
                if min_dist < self.args.dist_thres:
                    best_detection = person_detections[best_g_idx]
                    best_match_info = {'detection': best_detection, 'dist': min_dist}
        
        # 更新状态用于可视化和发送
        if best_match_info:
            best_detection = best_match_info['detection']
            self.last_tracked_bbox = best_detection['box']
            self.last_tracked_kpts = best_detection['keypoints']
            self.last_tracked_kpts_conf = best_detection['keypoints_conf']
            self.last_match_dist = best_match_info['dist']
            
            # 使用关键点计算人体中心
            body_center = calculate_body_center_from_keypoints(
                best_detection['keypoints'], 
                best_detection['keypoints_conf'], 
                best_detection['box']
            )
            
            size = (self.last_tracked_bbox[2] - self.last_tracked_bbox[0], 
                   self.last_tracked_bbox[3] - self.last_tracked_bbox[1])
            
            if self.current_depth_frame is not None:
                coords = calculate_3d_coordinates(self.current_depth_frame, body_center, size)
                if coords != (0,0,0):
                    self.last_coords = coords
                    
                    # 使用卡尔曼滤波器处理坐标
                    measurement = np.array([coords[0], coords[1], coords[2]])
                    
                    if not self.ekf.is_initialized():
                        # 初始化卡尔曼滤波器
                        self.ekf.initialize(measurement, current_time)
                        self.last_filtered_coords = coords
                        self.last_predicted_coords = coords
                        print(f"🎯 卡尔曼滤波器已初始化: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]")
                    else:
                        # 预测和更新
                        self.ekf.predict(current_time)
                        filtered_state = self.ekf.update(measurement)
                        self.last_filtered_coords = self.ekf.get_current_position()
                        self.last_predicted_coords = self.ekf.predict_future_position(0.5)  # 预测0.5秒后的位置

                        # 打印调试信息
                        velocity = self.ekf.get_current_velocity()
                        print(f"📍 原始: [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}] | "
                              f"滤波: [{self.last_filtered_coords[0]:.2f}, {self.last_filtered_coords[1]:.2f}, {self.last_filtered_coords[2]:.2f}] | "
                              f"速度: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")
                else:
                    self.last_coords = None
                    # 处理目标丢失情况
                    if self.ekf.is_initialized():
                        predicted_pos = self.ekf.handle_lost_target(current_time)
                        if predicted_pos is not None:
                            self.last_filtered_coords = predicted_pos
                            self.last_predicted_coords = self.ekf.predict_future_position(0.5)
                            print(f"🔍 目标丢失，使用预测位置: [{predicted_pos[0]:.2f}, {predicted_pos[1]:.2f}, {predicted_pos[2]:.2f}]")
                        else:
                            self.last_filtered_coords = None
                            self.last_predicted_coords = None
            else:
                self.last_coords = None
                self.last_filtered_coords = None
                self.last_predicted_coords = None
        else:
            self.last_tracked_bbox = None
            self.last_tracked_kpts = None
            self.last_tracked_kpts_conf = None
            self.last_coords = None
            
            # 处理目标丢失情况
            if self.ekf.is_initialized():
                predicted_pos = self.ekf.handle_lost_target(current_time)
                if predicted_pos is not None:
                    self.last_filtered_coords = predicted_pos
                    self.last_predicted_coords = self.ekf.predict_future_position(0.5)
                    print(f"🔍 目标丢失，使用预测位置: [{predicted_pos[0]:.2f}, {predicted_pos[1]:.2f}, {predicted_pos[2]:.2f}]")
                else:
                    self.last_filtered_coords = None
                    self.last_predicted_coords = None
                    self.ekf.reset()  # 重置滤波器
                    print("🔄 目标丢失时间过长，滤波器已重置")
            else:
                self.last_filtered_coords = None
                self.last_predicted_coords = None

        # 发送坐标 - 优先发送滤波后的坐标，其次是原始坐标，最后是 (0, 0, 0)
        coords_to_send = (0.0, 0.0, 0.0)
        if self.last_filtered_coords:
            coords_to_send = self.last_filtered_coords
        elif self.last_coords:
            coords_to_send = self.last_coords
        
        if self.grpc_client:
            self.grpc_client.send_target_coordinates(coords_to_send)
            
        # 导出坐标到文件 (用于ROS2集成) - 使用滤波后的坐标
        if self.coord_exporter:
            self.coord_exporter.export_coordinates(coords_to_send)

    def transition_to_tracking(self):
        self.state = 'TRACKING'
    def transition_to_idle(self):
        self.state = 'IDLE'
        self.query_feats = None
        self.captured_features = []
        # 重置卡尔曼滤波器
        self.ekf.reset()
        self.last_filtered_coords = None
        self.last_predicted_coords = None
        print("🔄 转换到待机状态，卡尔曼滤波器已重置")

    def create_visualization(self, frame):
        vis_frame = frame.copy()
        
        # 根据状态绘制不同的框和姿态
        if self.state == 'CAPTURING':
            detection = find_center_person(vis_frame, self.yolo_model)
            if detection: 
                draw_pose_on_person(vis_frame, detection, "Capturing...", (0, 165, 255))
        elif self.state == 'TRACKING':
            # 绘制所有检测到的人物（浅色）
            all_detections = detect_all_poses(vis_frame, self.yolo_model, self.args.conf_thres)
            for detection in all_detections:
                # 用浅色绘制所有检测到的人
                draw_skeleton(vis_frame, detection['keypoints'], detection['keypoints_conf'])
                plot_one_box(detection['box'], vis_frame, label="Person", color=(128, 128, 128))
            
            # 绘制跟踪目标（高亮）
            if self.last_tracked_bbox and self.last_tracked_kpts is not None:
                label = f"Target | Dist: {self.last_match_dist:.2f}"
                
                # 显示原始坐标
                if self.last_coords:
                    label += f' | Raw: {self.last_coords[0]:.1f}, {self.last_coords[1]:.1f}, {self.last_coords[2]:.1f}m'
                
                # 显示滤波后的坐标
                if self.last_filtered_coords:
                    label += f' | Filtered: {self.last_filtered_coords[0]:.1f}, {self.last_filtered_coords[1]:.1f}, {self.last_filtered_coords[2]:.1f}m'
                
                # 绘制跟踪目标的边界框
                plot_one_box(self.last_tracked_bbox, vis_frame, label=label, color=(0,255,0))
                
                # 创建伪检测对象来绘制骨架
                tracked_detection = {
                    'box': self.last_tracked_bbox,
                    'keypoints': self.last_tracked_kpts,
                    'keypoints_conf': self.last_tracked_kpts_conf
                }
                
                # 绘制跟踪目标的骨架和关键点（高亮显示）
                draw_skeleton(vis_frame, tracked_detection['keypoints'], tracked_detection['keypoints_conf'])
                draw_keypoints(vis_frame, tracked_detection['keypoints'], tracked_detection['keypoints_conf'])
                
                # 绘制人体中心点（特殊标记）
                body_center = calculate_body_center_from_keypoints(
                    tracked_detection['keypoints'], tracked_detection['keypoints_conf'], tracked_detection['box']
                )
                center_x, center_y = int(body_center[0]), int(body_center[1])
                cv2.circle(vis_frame, (center_x, center_y), 8, (0, 255, 255), -1)  # 黄色圆点
                cv2.circle(vis_frame, (center_x, center_y), 10, (0, 0, 0), 2)     # 黑色边框
                
                # 绘制预测位置的投影（如果有）
                if self.last_predicted_coords and self.current_depth_frame is not None:
                    # 将3D预测位置投影回图像坐标
                    fx, fy = 734.357, 734.629
                    cx, cy = 816.469, 642.979
                    
                    X_world, Y_world, Z_world = self.last_predicted_coords
                    Z_cam = X_world
                    X_cam = -Y_world
                    Y_cam = -Z_world
                    
                    if Z_cam > 0.3:  # 只有在合理距离内才绘制
                        u_pred = int(X_cam * fx / Z_cam + cx)
                        v_pred = int(Y_cam * fy / Z_cam + cy)
                        
                        # 确保坐标在图像范围内
                        if 0 <= u_pred < vis_frame.shape[1] and 0 <= v_pred < vis_frame.shape[0]:
                            cv2.circle(vis_frame, (u_pred, v_pred), 12, (255, 255, 0), 3)  # 青色预测圆圈
                            cv2.putText(vis_frame, "Pred", (u_pred-20, v_pred-15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 绘制固定的UI元素
        self.frame_count += 1
        if time.time() - self.start_time > 1:
            self.fps = self.frame_count / (time.time() - self.start_time)
            self.start_time = time.time()
            self.frame_count = 0
        
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, self.status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示卡尔曼滤波器状态
        if self.ekf.is_initialized():
            uncertainty = self.ekf.get_position_uncertainty()
            velocity = self.ekf.get_current_velocity()
            ekf_status = f"Enhanced EKF: Init | Unc: {uncertainty:.3f} | Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]"
            cv2.putText(vis_frame, ekf_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(vis_frame, "Enhanced EKF: Not Initialized", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # 显示关键点信息
        if self.state == 'TRACKING' and self.last_tracked_kpts is not None:
            valid_kpts = sum(1 for conf in self.last_tracked_kpts_conf if conf > 0.5)
            kpt_info = f"Keypoints: {valid_kpts}/17"
            cv2.putText(vis_frame, kpt_info, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return vis_frame

    def extract_gallery_features(self, frame, person_detections):
        """从姿态检测结果中提取特征"""
        gallery_locs, gallery_img_tensors = [], []
        for detection in person_detections:
            xmin, ymin, xmax, ymax = detection['box']
            crop_img = frame[ymin:ymax, xmin:xmax]
            if crop_img.size > 0:
                gallery_locs.append((xmin, ymin, xmax, ymax))
                crop_img_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                gallery_img_tensors.append(build_transforms(reidCfg)(crop_img_pil).unsqueeze(0))
        if not gallery_img_tensors: return None, None
        gallery_img = torch.cat(gallery_img_tensors, dim=0).to(self.device)
        with torch.no_grad():
            gallery_feats = self.reid_model(gallery_img)
            gallery_feats = F.normalize(gallery_feats, dim=1, p=2)
        return gallery_locs, gallery_feats

    def calculate_distance_matrix(self, gallery_feats):
        m, n = self.query_feats.shape[0], gallery_feats.shape[0]
        distmat = torch.pow(self.query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(self.query_feats, gallery_feats.t(), beta=1, alpha=-2)
        return distmat.cpu().numpy()

# ==============================================================================
# ROS2 节点和主程序
# ==============================================================================
class ImageSubscriberNode(Node):
    def __init__(self, args):
        super().__init__('reid_tracker_node')
        self.get_logger().info("=== ReID ROS2 跟踪节点启动 ===")
        
        self.bridge = CvBridge()
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5) if not args.no_viz else None
        self.stop_event = threading.Event()
        self.start_event = threading.Event()
        
        # 加载模型
        self.get_logger().info("正在加载模型...")
        try:
            self.yolo_model = YOLO(args.model_path)
            self.reid_model = build_model(reidCfg, num_classes=1501)
            self.reid_model.load_param(reidCfg.TEST.WEIGHT)
            self.reid_model.eval()
            self.get_logger().info("✓ 模型加载完成")
        except Exception as e:
            self.get_logger().error(f"❌ 模型加载失败: {e}")
            raise e

        # 初始化gRPC客户端
        self.grpc_client = TrackingGRPCClient(args.grpc_server) if not args.no_grpc else None
        
        # 启动处理线程
        self.processing_thread = ProcessingThread(
            self.frame_queue, self.result_queue, self.stop_event, 
            self.start_event, self.grpc_client, args, 
            self.yolo_model, self.reid_model
        )
        self.processing_thread.start()
        self.get_logger().info("✓ 后台处理线程已启动...")

        # 设置订阅器
        self.color_sub = message_filters.Subscriber(self, RosImage, '/odin1/image_undistorted')
        self.depth_sub = message_filters.Subscriber(self, RosImage, '/odin1/depth/image_raw')

        # 时间同步器
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1  # 允许0.1秒的时间戳差异
        )
        self.time_synchronizer.registerCallback(self.image_callback)
        self.get_logger().info("✓ 已订阅彩色和深度图像话题，等待同步消息...")

    def image_callback(self, color_msg, depth_msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # 深度图是32FC1格式，直接转换
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge转换错误: {e}")
            return
        
        # 将帧放入队列供处理线程使用
        if self.frame_queue.full():
            self.frame_queue.get_nowait() # 如果队列满了，丢弃旧的帧
        self.frame_queue.put((color_frame, depth_frame))

    def stop_all_threads(self):
        self.get_logger().info("正在停止所有线程...")
        self.stop_event.set()
        self.processing_thread.join(timeout=5)
        if self.processing_thread.is_alive():
            self.get_logger().warn("处理线程未能正常停止。")

def main(args):
    rclpy.init(args=None)
    
    try:
        reid_node = ImageSubscriberNode(args)
    except Exception as e:
        print(f"节点初始化失败: {e}")
        rclpy.shutdown()
        return

    if not args.no_viz:
        window_name = 'ReID ROS2 Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        while rclpy.ok():
            rclpy.spin_once(reid_node, timeout_sec=0.01)
            try:
                display_frame = reid_node.result_queue.get_nowait()
                cv2.imshow(window_name, display_frame)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("键盘 'R' 已按下，发送开始信号...")
                reid_node.start_event.set()
        
        cv2.destroyAllWindows()
    else:
        # 在无头模式下运行
        try:
            rclpy.spin(reid_node)
        except KeyboardInterrupt:
            pass

    # 清理
    reid_node.stop_all_threads()
    reid_node.destroy_node()
    rclpy.shutdown()
    print("程序已安全退出。")

def parse_args():
    parser = argparse.ArgumentParser(description='ROS2 ReID Auto Tracking with Pose Detection and Kalman Filter')
    parser.add_argument('--model-path', type=str, default='models/yolo11n-pose.pt', help='YOLOv11-Pose模型路径')
    parser.add_argument('--dist-thres', type=float, default=1.1, help='ReID距离阈值')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='YOLO检测置信度阈值')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (e.g., cpu, cuda:0)')
    parser.add_argument('--grpc-server', default='localhost:50051', help='gRPC服务器地址')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化界面')
    parser.add_argument('--no-grpc', action='store_true', help='禁用gRPC通信')
    parser.add_argument('--no-ros-export', action='store_true', help='禁用ROS2坐标导出')
    
    # 卡尔曼滤波器参数
    parser.add_argument('--ekf-process-noise', type=float, default=0.5, help='卡尔曼滤波器过程噪声标准差')
    parser.add_argument('--ekf-measurement-noise', type=float, default=3.0, help='卡尔曼滤波器测量噪声标准差')
    parser.add_argument('--ekf-velocity-std', type=float, default=0.5, help='卡尔曼滤波器初始速度不确定性标准差')
    parser.add_argument('--ekf-acceleration-std', type=float, default=0.3, help='卡尔曼滤波器初始加速度不确定性标准差')
    parser.add_argument('--ekf-angular-velocity-std', type=float, default=0.2, help='卡尔曼滤波器初始角速度不确定性标准差')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用的计算设备: {args.device}")
    
    # 显示卡尔曼滤波器配置信息
    print(f"🎯 卡尔曼滤波器配置: Enhanced EKF (匀加速运动模型)")
    print(f"   过程噪声: {args.ekf_process_noise}")
    print(f"   测量噪声: {args.ekf_measurement_noise}")
    print(f"   速度不确定性: {args.ekf_velocity_std}")
    print(f"   加速度不确定性: {args.ekf_acceleration_std}")
    print(f"   角速度不确定性: {args.ekf_angular_velocity_std}")
    
    main(args)
