# 文件名: track_reid_grpc_auto_viz.py
# 描述: 自动选择中心目标，由gRPC指令或键盘'R'键触发，进行10秒特征捕获后开始跟踪。
# 版本: v4.2 - 优化了可视化逻辑，确保跟踪框稳定显示。

import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import depthai as dai
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
# OAK相机与ReID核心逻辑
# ==============================================================================
def create_camera_pipeline():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)
    return pipeline

def calculate_3d_coordinates(depth_map, center_point, size=None):
    u, v = int(center_point[0]), int(center_point[1])
    height, width = depth_map.shape
    w, h = (10, 10) if size is None else size
    roi_size = max(5, int(min(w, h) * 0.1))
    x1, y1 = max(0, u - roi_size), max(0, v - roi_size)
    x2, y2 = min(width - 1, u + roi_size), min(height - 1, v + roi_size)
    if x1 >= x2 or y1 >= y2: return (0, 0, 0)
    depth_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
    valid_mask = (depth_roi > 300) & (depth_roi < 8000)
    if not np.any(valid_mask): return (0, 0, 0)
    median_depth = np.median(depth_roi[valid_mask])
    Z_cam = median_depth / 1000.0
    if Z_cam <= 0.3 or Z_cam > 15.0: return (0, 0, 0)
    fx, fy = 860.0, 860.0
    cx, cy = width / 2.0, height / 2.0
    try:
        X_cam = (u - cx) * Z_cam / fx
        Y_cam = (v - cy) * Z_cam / fy
        X_world = Z_cam
        Y_world = -X_cam
        Z_world = -Y_cam
    except ZeroDivisionError: return (0, 0, 0)
    if any(math.isnan(val) for val in (X_world, Y_world, Z_world)): return (0, 0, 0)
    return (X_world, Y_world, Z_world)

def detect_all_persons(frame, model, conf_thres=0.5):
    results = model.predict(source=frame, show=False, classes=[0], conf=conf_thres, verbose=False)
    boxes = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            if box.conf[0] > conf_thres:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].cpu().numpy())
                if (xmax - xmin) * (ymax - ymin) > 2000:
                    boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def find_center_person(frame, yolo_model):
    boxes = detect_all_persons(frame, yolo_model)
    if not boxes: return None
    frame_center_x, frame_center_y = frame.shape[1] / 2, frame.shape[0] / 2
    min_dist = float('inf')
    center_box = None
    for (xmin, ymin, xmax, ymax) in boxes:
        box_center_x = (xmin + xmax) / 2
        box_center_y = (ymin + ymax) / 2
        dist = math.sqrt((box_center_x - frame_center_x)**2 + (box_center_y - frame_center_y)**2)
        if dist < min_dist:
            min_dist = dist
            center_box = (xmin, ymin, xmax, ymax)
    return center_box


# ==============================================================================
# 多线程框架
# ==============================================================================
class CameraManager:
    def __init__(self, max_retries=3, retry_delay=3):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.device = None
        self.pipeline = create_camera_pipeline()
    def connect_camera(self):
        for attempt in range(self.max_retries):
            try:
                print(f"尝试连接OAK相机... (第 {attempt + 1}/{self.max_retries} 次)")
                self.device = dai.Device(self.pipeline)
                print("OAK相机连接成功！")
                return True
            except Exception as e:
                print(f"相机连接失败: {e}")
                if self.device: self.device.close()
                self.device = None
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return False
    def get_device(self): return self.device
    def close(self):
        if self.device: self.device.close()

class FrameCaptureThread(threading.Thread):
    def __init__(self, device, frame_queue):
        super().__init__()
        self.device = device
        self.frame_queue = frame_queue
        self.running = True
        self.q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    def run(self):
        while self.running:
            try:
                in_rgb = self.q_rgb.get()
                in_depth = self.q_depth.get()
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put((in_rgb.getCvFrame(), in_depth.getFrame()))
            except Exception as e:
                if self.running: print(f"相机线程错误: {e}")
                self.running = False
        print("相机线程已停止。")
    def stop(self):
        self.running = False

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
        
        # 可视化相关
        self.last_tracked_bbox = None
        self.last_match_dist = 0.0
        self.last_coords = None
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
                frame, depth_frame_raw = self.frame_queue.get(timeout=1)
                self.current_depth_frame = cv2.medianBlur(depth_frame_raw.astype(np.float32), 5).astype(np.uint16)
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
            self.status_message = "状态: 待机 (按R或等待gRPC指令)"
            if start_signal:
                self.transition_to_capturing(frame)
        
        elif self.state == 'CAPTURING':
            self.process_capturing(frame)

        elif self.state == 'TRACKING':
            self.process_tracking(frame)
            # 检查gRPC停止信号 (保持原有逻辑不变)
            if self.grpc_client and (time.time() - self.last_grpc_check_time > 1.0):
                self.last_grpc_check_time = time.time()
                is_active, _ = self.grpc_client.get_command_state()
                if not is_active and self.grpc_client.connected:
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
                timestamp = data.get('timestamp', 0)
                
                # 检查命令是否太旧（超过5秒）
                if time.time() - timestamp > 5.0:
                    self.ros_command_active = False
                    self.last_ros_command_active = False
                    return False
                
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
        if self.grpc_client and (time.time() - self.last_grpc_check_time > 1.0):
            self.last_grpc_check_time = time.time()
            is_active, _ = self.grpc_client.get_command_state()
            if is_active:
                self._started_by_ros = False
                print("收到gRPC开始指令，准备开始捕获...")
                return True
                
        # 3. 检查ROS2控制信号 (新增，不影响gRPC)
        # 先保存当前的ROS状态
        prev_ros_active = self.last_ros_command_active
        # 读取最新的ROS2状态
        current_ros_active = self.check_ros_control_signal()
        
        # 只有在从非激活状态变为激活状态时才触发开始信号
        if current_ros_active and not prev_ros_active and self.state == 'IDLE':
            self._started_by_ros = True
            print("检测到ROS2信号从0变为1，准备开始捕获...")
            return True
            
        return False

    def transition_to_capturing(self, frame):
        initial_bbox = find_center_person(frame, self.yolo_model)
        if initial_bbox is None:
            print("启动失败：画面中央未检测到目标。")
            return
        self.state = 'CAPTURING'
        self.captured_features = []
        self.capture_start_time = time.time()
        self.last_capture_time = time.time() - 1.9
        self.status_message = "collecting... (0/5)"
        print(f"目标锁定：{initial_bbox}。开始10秒特征捕获...")

    def process_capturing(self, frame):
        time_elapsed = time.time() - self.capture_start_time
        if time_elapsed > 10.0:
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
        if len(self.captured_features) < 5 and (time.time() - self.last_capture_time) > 2.0:
            bbox = find_center_person(frame, self.yolo_model)
            if bbox:
                (xmin, ymin, xmax, ymax) = bbox
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
        person_boxes = detect_all_persons(frame, self.yolo_model, self.args.conf_thres)
        best_match_info = None
        if person_boxes:
            gallery_locs, gallery_feats = self.extract_gallery_features(frame, person_boxes)
            if gallery_feats is not None:
                distmat = self.calculate_distance_matrix(gallery_feats)
                best_g_idx = np.argmin(distmat[0])
                min_dist = distmat[0, best_g_idx]
                if min_dist < self.args.dist_thres:
                    best_match_info = {'bbox': gallery_locs[best_g_idx], 'dist': min_dist}
        
        # 更新状态用于可视化和发送
        if best_match_info:
            self.last_tracked_bbox = best_match_info['bbox']
            self.last_match_dist = best_match_info['dist']
            center = ((self.last_tracked_bbox[0] + self.last_tracked_bbox[2]) / 2, (self.last_tracked_bbox[1] + self.last_tracked_bbox[3]) / 2)
            size = (self.last_tracked_bbox[2] - self.last_tracked_bbox[0], self.last_tracked_bbox[3] - self.last_tracked_bbox[1])
            if self.current_depth_frame is not None:
                coords = calculate_3d_coordinates(self.current_depth_frame, center, size)
                self.last_coords = coords if coords != (0,0,0) else None
            else:
                self.last_coords = None
        else:
            self.last_tracked_bbox = None
            self.last_coords = None

        # 发送坐标 - 如果没有检测到目标，发送 (0, 0, 0)
        coords_to_send = self.last_coords if self.last_coords else (0.0, 0.0, 0.0)
        
        if self.grpc_client:
            self.grpc_client.send_target_coordinates(coords_to_send)
            
        # 导出坐标到文件 (用于ROS2集成)
        if self.coord_exporter:
            self.coord_exporter.export_coordinates(coords_to_send)

    def transition_to_tracking(self):
        self.state = 'TRACKING'
    def transition_to_idle(self):
        self.state = 'IDLE'
        self.query_feats = None
        self.captured_features = []

    def create_visualization(self, frame):
        vis_frame = frame.copy()
        
        # 根据状态绘制不同的框
        if self.state == 'CAPTURING':
            bbox = find_center_person(vis_frame, self.yolo_model)
            if bbox: plot_one_box(bbox, vis_frame, label='Capturing...', color=(0, 165, 255))
        elif self.state == 'TRACKING' and self.last_tracked_bbox:
            label = f"Target | Dist: {self.last_match_dist:.2f}"
            if self.last_coords:
                label += f' | Coords: {self.last_coords[0]:.1f}, {self.last_coords[1]:.1f}, {self.last_coords[2]:.1f}m'
            plot_one_box(self.last_tracked_bbox, vis_frame, label=label, color=(0,255,0))
        
        # 绘制固定的UI元素
        self.frame_count += 1
        if time.time() - self.start_time > 1:
            self.fps = self.frame_count / (time.time() - self.start_time)
            self.start_time = time.time()
            self.frame_count = 0
        
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, self.status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return vis_frame

    def extract_gallery_features(self, frame, person_boxes):
        gallery_locs, gallery_img_tensors = [], []
        for xmin, ymin, xmax, ymax in person_boxes:
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
# 主程序
# ==============================================================================
def main(args):
    print("=== OAK ReID 自动指令跟踪系统 ===")
    
    camera_manager = CameraManager()
    if not camera_manager.connect_camera(): return
    
    device = torch.device(args.device)
    try:
        print("正在加载模型...")
        yolo_model = YOLO(args.model_path)
        reid_model = build_model(reidCfg, num_classes=1501)
        reid_model.load_param(reidCfg.TEST.WEIGHT)
        reid_model.eval()
        print("✓ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        camera_manager.close()
        return

    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2) if not args.no_viz else None
    stop_event = threading.Event()
    start_event = threading.Event()
    
    grpc_client = TrackingGRPCClient(args.grpc_server) if not args.no_grpc else None

    capture_thread = FrameCaptureThread(camera_manager.get_device(), frame_queue)
    processing_thread = ProcessingThread(frame_queue, result_queue, stop_event, start_event, grpc_client, args, yolo_model, reid_model)

    capture_thread.start()
    processing_thread.start()
    print("✓ 后台处理线程已启动...")

    if not args.no_viz:
        window_name = 'OAK ReID Auto Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        while not stop_event.is_set():
            try:
                display_frame = result_queue.get(timeout=2)
                cv2.imshow(window_name, display_frame)
            except queue.Empty:
                if not processing_thread.is_alive():
                    print("❌ 处理线程已意外终止。")
                    break
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): stop_event.set()
            elif key == ord('r'):
                print("键盘 'R' 已按下，发送开始信号...")
                start_event.set()

        cv2.destroyAllWindows()
    else:
        try:
            while not stop_event.is_set(): time.sleep(1)
        except KeyboardInterrupt:
            stop_event.set()

    print("正在停止所有线程...")
    stop_event.set()
    capture_thread.stop()
    capture_thread.join(timeout=2)
    processing_thread.join(timeout=5)
    camera_manager.close()
    print("程序已安全退出。")

def parse_args():
    parser = argparse.ArgumentParser(description='OAK ReID Auto Tracking with gRPC')
    parser.add_argument('--model-path', type=str, default='yolo11n.pt', help='YOLOv8模型路径')
    parser.add_argument('--dist-thres', type=float, default=1.2, help='ReID距离阈值')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='YOLO检测置信度阈值')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (e.g., cpu, cuda:0)')
    parser.add_argument('--grpc-server', default='localhost:50051', help='gRPC服务器地址')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化界面')
    parser.add_argument('--no-grpc', action='store_true', help='禁用gRPC通信')
    parser.add_argument('--no-ros-export', action='store_true', help='禁用ROS2坐标导出')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用的计算设备: {args.device}")
    main(args)