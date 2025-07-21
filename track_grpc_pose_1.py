#REID+姿态+前10s无重复特征+gRPC通信
# 
# 性能优化版本 - 单目标跟踪系统
# 
# [修改V4] 修复 NameError 导致的线程崩溃问题
#

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import math
import depthai as dai
import onnxruntime
import sys
import os
import pickle
import threading
import queue
import grpc
import argparse

# 导入生成的gRPC模块
try:
    import tracking_pb2
    import tracking_pb2_grpc
except ImportError:
    print("警告: 未找到gRPC模块，gRPC通信功能将被禁用")
    print("请运行: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. tracking.proto")
    tracking_pb2 = None
    tracking_pb2_grpc = None

# 全局变量：特征存储路径
FEATURE_STORAGE_PATH = "target_features.pkl"

# 确保清理旧特征文件
if os.path.exists(FEATURE_STORAGE_PATH):
    os.remove(FEATURE_STORAGE_PATH)
    print(f"已清理旧特征文件: {FEATURE_STORAGE_PATH}")

# 初始化ReID特征提取器
class ReIDFeatureExtractor:
    def __init__(self, model_path="reidmodel_fp16.onnx"):
        try:
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = onnxruntime.InferenceSession(
                model_path,
                sess_options,
                providers=['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            self.input_size = (input_shape[3], input_shape[2])
            print(f"模型输入形状: {input_shape}, 调整输入尺寸为: {self.input_size}")
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            print("ReID特征提取器初始化完成，使用ONNX Runtime")
        except Exception as e:
            print(f"ReID模型加载失败: {e}")
            self.session = None
            print("将使用颜色直方图作为替代特征提取方法")

    def extract_features(self, image):
        if self.session:
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(image_rgb, self.input_size).astype(np.float32)
                normalized = (resized / 255.0 - self.mean) / self.std
                input_data = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
                if input_data.dtype != np.float16:
                    input_data = input_data.astype(np.float16)
                features = self.session.run(None, {self.input_name: input_data})[0]
                norm = np.linalg.norm(features)
                return features / norm if norm > 0 else features
            except Exception:
                return self.fallback_feature_extractor(image)
        else:
            return self.fallback_feature_extractor(image)

    def fallback_feature_extractor(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

# 初始化ReID特征提取器
reid_extractor = ReIDFeatureExtractor()

# 姿态特征提取与相似度计算
def extract_pose_features(keypoints):
    if keypoints is None or len(keypoints) < 17: return None
    kpts = keypoints.copy()
    valid_mask = (kpts[:, 0] > 0) & (kpts[:, 1] > 0)
    valid_kpts = kpts[valid_mask]
    if len(valid_kpts) < 5: return None
    left_shoulder, right_shoulder, left_hip, right_hip = 5, 6, 11, 12
    center_points = []
    if valid_mask[left_shoulder] and valid_mask[right_shoulder]: center_points.extend([kpts[left_shoulder], kpts[right_shoulder]])
    if valid_mask[left_hip] and valid_mask[right_hip]: center_points.extend([kpts[left_hip], kpts[right_hip]])
    center = np.mean(center_points, axis=0) if len(center_points) > 0 else np.mean(valid_kpts, axis=0)
    distances = np.linalg.norm(valid_kpts - center, axis=1)
    scale = np.max(distances) if len(distances) > 0 and np.max(distances) > 0 else 1.0
    features = []
    for i in range(17):
        if valid_mask[i]: features.extend(((kpts[i] - center) / scale).tolist())
        else: features.extend([0.0, 0.0])
    return np.array(features)

def pose_similarity(feat1, feat2):
    if feat1 is None or feat2 is None: return 0.0
    norm1, norm2 = np.linalg.norm(feat1), np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return max(0.0, min(1.0, np.dot(feat1, feat2) / (norm1 * norm2)))

def draw_pose_keypoints(img, keypoints, color=(0, 255, 255)):
    if keypoints is None or len(keypoints) < 17: return img
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0: cv2.circle(img, (int(x), int(y)), 3, color, -1)
    for connection in skeleton:
        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
        if pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0:
            pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
            pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
            cv2.line(img, pt1, pt2, color, 2)
    return img

# gRPC客户端类
class TrackingGRPCClient:
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.connected = False
        self.coordinate_queue = queue.Queue(maxsize=100)
        self.stream_thread = None
        self.streaming = False
        self.last_coordinate_time = 0
        self.coordinate_interval = 0.1

    def connect(self):
        if tracking_pb2 is None: return False
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
                    yield self.coordinate_queue.get(timeout=1.0)
                except queue.Empty:
                    yield tracking_pb2.CoordinateData(active=False, timestamp=time.time())
        def stream_worker():
            try:
                self.streaming = True
                response = self.stub.SendCoordinates(coordinate_generator())
                # print(f"坐标流传输结果: {response.message}")
            except Exception: pass
            finally: self.streaming = False
        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()
        print("📡 坐标流传输已启动")

    def disconnect(self):
        self.streaming = False
        if self.stream_thread: self.stream_thread.join(timeout=2)
        if self.channel: self.channel.close()
        self.connected = False
        print("gRPC客户端已断开连接")

    def send_target_coordinates(self, target_state):
        if not self.connected or not self.streaming: return False
        current_time = time.time()
        if current_time - self.last_coordinate_time < self.coordinate_interval: return True
        try:
            msg = None
            if target_state and target_state.active and target_state.world_position and target_state.lost_frame_count == 0:
                X, Y, Z = target_state.world_position
                msg = tracking_pb2.CoordinateData(
                    x=float(X), y=float(Y), z=float(Z),
                    distance=float(target_state.distance), yaw=float(target_state.yaw), pitch=float(target_state.pitch),
                    target_id=int(target_state.target_id), confidence=float(target_state.confidence),
                    active=True, timestamp=current_time
                )
            else:
                msg = tracking_pb2.CoordinateData(active=False, timestamp=current_time)
            
            if self.coordinate_queue.full(): self.coordinate_queue.get_nowait()
            self.coordinate_queue.put_nowait(msg)
            self.last_coordinate_time = current_time
            return True
        except Exception: return False
    
    def get_follow_commands(self):
        if not self.connected: return None
        try:
            response = self.stub.GetTrackingStatus(tracking_pb2.Empty())
            if response.is_active and response.target_id > 0:
                return [{'action': 'follow', 'target_id': response.target_id}]
            elif not response.is_active:
                return [{'action': 'stop_follow'}]
            return None
        except Exception: return None

# 特征存储管理器
class FeatureStorage:
    def __init__(self, file_path=FEATURE_STORAGE_PATH):
        self.file_path = file_path
        self.features = {}
    def save_features(self):
        try:
            with open(self.file_path, 'wb') as f: pickle.dump(self.features, f)
        except Exception as e: print(f"保存特征失败: {e}")
    def add_feature(self, target_id, feature):
        self.features.clear()
        self.features[target_id] = [feature]
    def clear_all_features(self):
        self.features.clear()
        if os.path.exists(self.file_path):
            try: os.remove(self.file_path)
            except Exception as e: print(f"删除特征文件失败: {e}")

# 相机管道与管理器
def create_pipeline():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(45)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)
    return pipeline

class CameraManager:
    def __init__(self, max_retries=5, retry_delay=3):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.device = None
    def connect_camera(self):
        for attempt in range(self.max_retries):
            try:
                print(f"尝试连接OAK相机... (第 {attempt + 1}/{self.max_retries} 次)")
                self.device = dai.Device(create_pipeline())
                print("OAK相机连接成功！")
                return True
            except Exception as e:
                print(f"相机连接失败 (第 {attempt + 1} 次): {e}")
                if self.device: self.device.close()
                self.device = None
                if attempt < self.max_retries - 1: time.sleep(self.retry_delay)
                else: return False
        return False
    def get_device(self): return self.device
    def close(self):
        if self.device: self.device.close()

# 目标状态类
class TargetState:
    def __init__(self, target_id):
        self.target_id = target_id
        self.position = None; self.velocity = (0, 0); self.size = None; self.confidence = 0.0
        self.timestamp = time.time(); self.last_seen = time.time(); self.last_update_time = time.time()
        self.world_position = None; self.distance = 0.0; self.yaw = 0.0; self.pitch = 0.0
        self.reid_features = deque(maxlen=20); self.stable_features = deque(maxlen=5)
        self.initial_feature = None; self.full_body_feature = None
        self.lock_strength = 1.0; self.stability = 0.0
        self.consecutive_frames = 0; self.lost_frame_count = 0; self.active = True
        self.pose_landmarks = None; self.pose_score = 0.0
        self.pose_features = deque(maxlen=10); self.stable_pose_features = deque(maxlen=3)
        self.initial_pose_feature = None
        self.kalman = self.init_kalman_filter()
    def init_kalman_filter(self):
        kalman = cv2.KalmanFilter(6, 2)
        kalman.transitionMatrix = np.array([[1,0,1,0,0.5,0],[0,1,0,1,0,0.5],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]], np.float32)
        kalman.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]], np.float32)
        kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        kalman.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
        return kalman
    def update_reid_features(self, frame, box, is_full_body=False):
        x1, y1, x2, y2 = map(int, box); expand = 10
        x1,y1,x2,y2 = max(0,x1-expand),max(0,y1-expand),min(frame.shape[1],x2+expand),min(frame.shape[0],y2+expand)
        if x1 >= x2 or y1 >= y2: return
        target_roi = frame[y1:y2, x1:x2]
        if target_roi.size == 0: return
        feature = reid_extractor.extract_features(target_roi)
        if self.initial_feature is None:
            self.initial_feature = feature
            self.stable_features.append(feature)
            if is_full_body:
                self.full_body_feature = feature
                # Note: This is where the global feature_storage was used before.
                # Now the manager will handle this if needed.
        if self.stability > 0.6: self.reid_features.append(feature)
        if self.stability > 0.8 and len(self.stable_features) < 5: self.stable_features.append(feature)
    def compare_signature(self, frame, box):
        # 修复 numpy 数组布尔歧义
        no_init = self.initial_feature is None
        no_stable = not self.stable_features or all(f is None for f in self.stable_features)
        no_full = self.full_body_feature is None
        if no_init and no_stable and no_full:
            return 0.0
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        candidate_roi = frame[y1:y2, x1:x2]
        if candidate_roi.size == 0:
            return 0.0
        candidate_feature = reid_extractor.extract_features(candidate_roi)
        max_similarity = 0.0
        features_to_check = list(self.reid_features) + list(self.stable_features)
        if self.initial_feature is not None:
            features_to_check.append(self.initial_feature)
        if self.full_body_feature is not None:
            features_to_check.append(self.full_body_feature)
        for feature in features_to_check:
            try:
                similarity = np.dot(feature.flatten(), candidate_feature.flatten()) / (np.linalg.norm(feature) * np.linalg.norm(candidate_feature) + 1e-10)
                if similarity > max_similarity:
                    max_similarity = similarity
            except:
                continue
        return max(0.0, min(1.0, max_similarity))
    def update_pose_features(self, keypoints):
        pose_feat = extract_pose_features(keypoints)
        if pose_feat is not None:
            self.pose_features.append(pose_feat)
            if self.initial_pose_feature is None: self.initial_pose_feature = pose_feat
            if self.stability > 0.8: self.stable_pose_features.append(pose_feat)
        if keypoints is not None:
            valid_kpts = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
            self.pose_score = len(valid_kpts) / 17.0 if len(keypoints) >= 17 else 0.0
            self.pose_landmarks = keypoints
    def compare_pose(self, keypoints):
        if not (self.initial_pose_feature is not None or len(self.stable_pose_features) > 0):
            return 0.0
        candidate_feature = extract_pose_features(keypoints)
        if candidate_feature is None:
            return 0.0
        max_similarity = 0.0
        if self.initial_pose_feature is not None:
            max_similarity = max(max_similarity, pose_similarity(self.initial_pose_feature, candidate_feature))
        for feature in self.stable_pose_features:
            max_similarity = max(max_similarity, pose_similarity(feature, candidate_feature))
        return max_similarity
    def update(self, x, y, w, h, conf, depth_map=None):
        current_time = time.time(); self.last_update_time = current_time
        self.consecutive_frames += 1
        self.stability = min(1.0, self.consecutive_frames / 30.0)
        self.lock_strength = min(1.0, self.lock_strength + 0.05)
        self.position = (x, y); self.size = (w, h); self.confidence = conf; self.timestamp = current_time
        self.last_seen = current_time; self.lost_frame_count = 0
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman.correct(measurement)
        self.world_position = calculate_3d_coordinates(depth_map, (x, y))
        if self.world_position and self.world_position != (0,0,0):
            X, Y, Z = self.world_position
            if Z > 0.001:
                self.distance = math.sqrt(X**2 + Y**2 + Z**2)
                self.yaw = math.atan2(X, Z)
                self.pitch = math.atan2(-Y, math.sqrt(X**2+Z**2))
    def predict_next_position(self):
        prediction = self.kalman.predict()
        return prediction[0][0], prediction[1][0]
    def mark_lost(self):
        self.lost_frame_count += 1
        if self.lost_frame_count < 10: self.lock_strength = max(0.3, self.lock_strength - 0.05)
        self.consecutive_frames = 0; self.stability = 0.0; self.pose_landmarks = None
        if self.lost_frame_count > 150: self.active = False # 5 seconds at 30fps

# 3D坐标计算
def calculate_3d_coordinates(depth_map, center_point):
    u, v = int(center_point[0]), int(center_point[1])
    if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]): return (0,0,0)
    
    y1, y2 = max(0, v - 5), min(depth_map.shape[0], v + 5)
    x1, x2 = max(0, u - 5), min(depth_map.shape[1], u + 5)
    depth_roi = depth_map[y1:y2, x1:x2]
    valid_depths = depth_roi[(depth_roi > 300) & (depth_roi < 8000)]
    if valid_depths.size == 0: return (0,0,0)
    
    median_depth = np.median(valid_depths)
    Z_cam = median_depth / 1000.0
    if Z_cam <= 0.3 or Z_cam > 15.0: return (0,0,0)

    fx, fy, cx, cy = 430.0, 430.0, 320.0, 240.0
    X_cam = (u - cx) * Z_cam / fx
    Y_cam = (v - cy) * Z_cam / fy
    return (Z_cam, -X_cam, -Y_cam)

# 跟踪管理器
class MultiTargetManager:
    # [FIX] Accept feature_storage as a dependency
    def __init__(self, feature_storage):
        self.targets = {}
        self.next_id = 0
        self.target_class = "person"
        self.follow_target_id = None
        self.follow_only_mode = False
        self.is_awaiting_first_target = False
        self.feature_storage = feature_storage # [FIX] Store the reference

    def get_new_target_id(self):
        self.next_id += 1
        return self.next_id

    def reset(self):
        self.targets.clear()
        self.next_id = 0
        self.follow_target_id = None
        self.follow_only_mode = False
        self.is_awaiting_first_target = True
        self.feature_storage.clear_all_features() # [FIX] Use the instance reference
        print("🔄 跟踪器已重置，待命捕获新目标...")

    def create_and_follow_target(self, frame, box, cls_id, conf, depth_map, keypoints=None):
        target_id = self.get_new_target_id()
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        bbox_w, bbox_h = abs(x2 - x1), abs(y2 - y1)
        
        new_target = TargetState(target_id)
        new_target.update(center_x, center_y, bbox_w, bbox_h, conf, depth_map)
        
        # Extract and store the primary feature
        feature = new_target.update_reid_features(frame, box, is_full_body=True)
        if new_target.full_body_feature is not None:
             self.feature_storage.add_feature(target_id, new_target.full_body_feature)

        if keypoints is not None:
            new_target.update_pose_features(keypoints)
        
        self.targets[target_id] = new_target
        self.follow_target_id = target_id
        self.follow_only_mode = True
        self.is_awaiting_first_target = False
        print(f"🎯 成功捕获并锁定第一个目标 ID: {target_id}。进入单目标跟踪模式。")

    def find_best_match(self, frame, boxes, cls_ids, confs, depth_map, keypoints_data):
        matches = []
        unmatched_detections = list(range(len(boxes)))

        if self.follow_only_mode and self.follow_target_id in self.targets:
            target = self.targets[self.follow_target_id]
            best_match_score, best_detection_idx, best_reid_sim = 0.0, -1, 0.0

            for i in unmatched_detections:
                if model.names[cls_ids[i]] != self.target_class:
                    continue
                reid_sim = target.compare_signature(frame, boxes[i])
                pose_sim = target.compare_pose(keypoints_data[i]) if keypoints_data is not None and i < len(keypoints_data) else 0.0
                px, py = target.predict_next_position()
                cx, cy = (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2
                dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                pos_score = max(0, 1 - dist / 300)
                match_score = (0.7 * reid_sim) + (0.2 * pose_sim) + (0.1 * pos_score)
                if match_score > best_match_score:
                    best_match_score, best_detection_idx, best_reid_sim = match_score, i, reid_sim

            if best_detection_idx != -1:
                is_accepted = False
                if target.lost_frame_count > 0: # Re-acquisition Mode
                    if best_match_score > 0.68 and best_reid_sim > 0.65: is_accepted = True
                else: # Sustained Tracking Mode
                    if best_match_score > min(0.85, 0.70 + target.stability * 0.2) and best_reid_sim > 0.68: is_accepted = True
                
                if is_accepted:
                    matches.append((self.follow_target_id, best_detection_idx, best_match_score))
                    unmatched_detections.remove(best_detection_idx)
            
        return matches, unmatched_detections
    
    def process_detections(self, frame, boxes, cls_ids, confs, depth_map, keypoints_data):
        for target in self.targets.values():
            if time.time() - target.last_seen > 0.1: target.mark_lost()
        
        inactive_ids = [tid for tid, t in self.targets.items() if not t.active]
        for tid in inactive_ids: 
            if tid in self.targets:
                del self.targets[tid]

        if self.is_awaiting_first_target and len(boxes) > 0:
            best_new_idx = -1; max_conf = 0.0
            for i in range(len(boxes)):
                if model.names[cls_ids[i]] == self.target_class and confs[i] > max_conf:
                    max_conf = confs[i]; best_new_idx = i
            
            if best_new_idx != -1:
                keypoints_for_target = keypoints_data[best_new_idx] if keypoints_data is not None and best_new_idx < len(keypoints_data) else None
                self.create_and_follow_target(frame, boxes[best_new_idx], cls_ids[best_new_idx], confs[best_new_idx], depth_map, keypoints_for_target)
                return

        matches, unmatched_detections = self.find_best_match(frame, boxes, cls_ids, confs, depth_map, keypoints_data)
        
        for target_id, detection_idx, _ in matches:
            target = self.targets[target_id]
            x1, y1, x2, y2 = boxes[detection_idx]
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1), abs(y2 - y1)
            target.update(cx, cy, w, h, confs[detection_idx], depth_map)
            if keypoints_data is not None and detection_idx < len(keypoints_data):
                target.update_pose_features(keypoints_data[detection_idx])
            target.update_reid_features(frame, boxes[detection_idx])
            
    def get_follow_target(self):
        return self.targets.get(self.follow_target_id)

# 线程类
class FrameCaptureThread(threading.Thread):
    def __init__(self, device, frame_queue):
        super().__init__(daemon=True)
        self.device = device; self.frame_queue = frame_queue; self.running = True
    def run(self):
        q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        while self.running:
            try:
                in_rgb = q_rgb.get()
                in_depth = q_depth.get()
                if in_rgb is not None and in_depth is not None:
                    frame, depth_frame = in_rgb.getCvFrame(), in_depth.getFrame()
                    if self.frame_queue.full(): self.frame_queue.get_nowait()
                    self.frame_queue.put((frame, depth_frame))
                else:
                    # Give CPU a break if no new frames
                    time.sleep(0.001)
            except RuntimeError as e:
                print(f"相机捕获线程错误: {e}. 线程将停止。")
                self.running = False
        print("相机线程已停止。")
    def stop(self): self.running = False

class ProcessingThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, stop_event, grpc_server, enable_viz):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue; self.result_queue = result_queue; self.stop_event = stop_event
        self.enable_visualization = enable_viz
        self.feature_storage = FeatureStorage() # [FIX] Create feature storage here
        self.target_manager = MultiTargetManager(self.feature_storage) # [FIX] Pass it to the manager
        self.grpc_client = TrackingGRPCClient(grpc_server)
        self.fps = 0; self.frame_count = 0; self.start_time = time.time()

    def run(self):
        global model
        self.grpc_client.connect()
        try:
            model = YOLO('yolov8n-pose.pt')
            print("YOLO模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}"); self.stop_event.set(); return

        while not self.stop_event.is_set():
            try: frame, depth_frame = self.frame_queue.get(timeout=1)
            except queue.Empty: continue

            self.handle_grpc_commands()
            
            results = model.track(frame, persist=True, verbose=False, conf=0.5, iou=0.6)
            
            boxes, cls_ids, confs, keypoints = [], [], [], None
            if results and results[0].boxes:
                res_boxes = results[0].boxes
                if len(res_boxes) > 0:
                    boxes = res_boxes.xyxy.cpu().numpy()
                    cls_ids = res_boxes.cls.cpu().numpy().astype(int)
                    confs = res_boxes.conf.cpu().numpy()
                    if hasattr(results[0], 'keypoints') and results[0].keypoints is not None and results[0].keypoints.xy is not None:
                        keypoints = results[0].keypoints.xy.cpu().numpy()
            
            self.target_manager.process_detections(frame, boxes, cls_ids, confs, depth_frame, keypoints)
            
            self.grpc_client.send_target_coordinates(self.target_manager.get_follow_target())
            
            if self.enable_visualization: self.draw_visualizations(frame)

            self.frame_count += 1
            if time.time() - self.start_time >= 1.0:
                self.fps = self.frame_count / (time.time() - self.start_time)
                self.start_time, self.frame_count = time.time(), 0
        
        self.grpc_client.disconnect()
        print("处理线程已停止。")

    def handle_grpc_commands(self):
        commands = self.grpc_client.get_follow_commands()
        if commands:
            for cmd in commands:
                if cmd['action'] == 'follow':
                    if not self.target_manager.follow_only_mode and not self.target_manager.is_awaiting_first_target:
                        self.target_manager.reset()
                elif cmd['action'] == 'stop_follow':
                    if self.target_manager.follow_only_mode or self.target_manager.is_awaiting_first_target:
                        # Reset and cancel the await state
                        self.target_manager.reset()
                        self.target_manager.is_awaiting_first_target = False

    def draw_visualizations(self, frame):
        vis_frame = frame.copy()
        follow_target = self.target_manager.get_follow_target()
        if follow_target and follow_target.position:
            w, h = follow_target.size; x, y = follow_target.position
            x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
            color = (0, 0, 255) if follow_target.lost_frame_count == 0 else (0, 165, 255)
            status = 'FOLLOW' if follow_target.lost_frame_count == 0 else 'LOST'
            self.draw_simple_box(vis_frame, [x1, y1, x2, y2], follow_target.target_id, follow_target.confidence, color, status, follow_target)

        self.draw_info(vis_frame, follow_target)
        if self.result_queue.full(): self.result_queue.get_nowait()
        self.result_queue.put(vis_frame)

    def draw_simple_box(self, vis_frame, bbox, tid, conf, color, status, target):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        if target and target.pose_landmarks is not None and target.pose_score > 0.5:
            draw_pose_keypoints(vis_frame, target.pose_landmarks, (0, 255, 0))
        label = f"ID{tid} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_frame, (x1, y1 - h - 10), (x1 + w, y1), (0,0,0), -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def draw_info(self, vis_frame, target):
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.target_manager.is_awaiting_first_target:
            cv2.putText(vis_frame, "AWAITING TARGET...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        elif target:
            status = "ACTIVE" if target.lost_frame_count == 0 else f"LOST({target.lost_frame_count}f)"
            color = (0, 255, 0) if status == "ACTIVE" else (0, 165, 255)
            cv2.putText(vis_frame, f"ID:{target.target_id} [{status}]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if target.distance > 0:
                cv2.putText(vis_frame, f"Dist:{target.distance:.2f}m Yaw:{math.degrees(target.yaw):.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
             cv2.putText(vis_frame, "No Target", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

# 主程序
def main(grpc_server, enable_visualization):
    print("=== OAK单目标跟踪系统 (V4 - 稳定版) ===")
    camera_manager = CameraManager()
    if not camera_manager.connect_camera(): return False
    
    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2) if enable_visualization else None
    stop_event = threading.Event()
    
    # Pass stop_event to threads for cleaner shutdown
    capture_thread = FrameCaptureThread(camera_manager.get_device(), frame_queue)
    processing_thread = ProcessingThread(frame_queue, result_queue, stop_event, grpc_server, enable_visualization)
    
    capture_thread.start()
    processing_thread.start()

    if enable_visualization:
        window_name = 'OAK单目标跟踪系统 - V4'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while not stop_event.is_set():
            try:
                display_frame = result_queue.get(timeout=1)
                cv2.imshow(window_name, display_frame)
            except queue.Empty:
                if not processing_thread.is_alive(): 
                    print("处理线程已停止，退出可视化。")
                    break
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                stop_event.set()
        cv2.destroyAllWindows()
    else:
        try:
            while not stop_event.is_set() and processing_thread.is_alive(): 
                time.sleep(1)
        except KeyboardInterrupt: 
            print("收到退出信号...")
            stop_event.set()
        
    print("正在停止...")
    stop_event.set()
    if capture_thread.is_alive(): capture_thread.stop() # Signal capture thread to stop
    capture_thread.join(timeout=2)
    if processing_thread.is_alive(): processing_thread.join(timeout=5)
    camera_manager.close()
    print("程序已退出。")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OAK单目标跟踪系统 - V4')
    parser.add_argument('--grpc-server', default='localhost:50051', help='gRPC服务器地址')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化界面')
    args = parser.parse_args()
    main(args.grpc_server, not args.no_viz)
    sys.exit(0)