#REID+姿态+前10s无重复特征
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import time
import math
import depthai as dai
import random
import onnxruntime
import heapq
import sys
import os
import pickle
import threading
import queue

# 全局配置：是否启用可视化（设为False可完全禁用可视化界面）
ENABLE_VISUALIZATION = True  # 设为False时将以无界面模式运行

# 全局变量：特征存储路径
FEATURE_STORAGE_PATH = "target_features.pkl"

# 确保清理旧特征文件
if os.path.exists(FEATURE_STORAGE_PATH):
    os.remove(FEATURE_STORAGE_PATH)
    print(f"已清理旧特征文件: {FEATURE_STORAGE_PATH}")

# 初始化ReID特征提取器
class ReIDFeatureExtractor:
    def __init__(self, model_path="fast-reid_model.onnx"):
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

                if input_data.dtype != np.float32:
                    input_data = input_data.astype(np.float32)

                features = self.session.run(None, {self.input_name: input_data})[0]
                norm = np.linalg.norm(features)
                return features / norm if norm > 0 else features
            except Exception as e:
                print(f"ReID特征提取失败: {e}")
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


# 特征存储管理器
class FeatureStorage:
    def __init__(self, file_path=FEATURE_STORAGE_PATH):
        self.file_path = file_path
        self.features = {}
        # 不加载旧特征，确保每次运行从零开始

    def save_features(self):
        """保存特征到文件"""
        try:
            with open(self.file_path, 'wb') as f:
                pickle.dump(self.features, f)
            print(f"已保存{len(self.features)}个目标的特征到{self.file_path}")
        except Exception as e:
            print(f"保存特征失败: {e}")

    def add_feature(self, target_id, feature):
        """添加或更新目标特征"""
        if target_id not in self.features:
            self.features[target_id] = []

        # 添加新特征并保留最新的10个
        self.features[target_id].append(feature)
        if len(self.features[target_id]) > 10:
            self.features[target_id] = self.features[target_id][-10:]

    def get_features(self, target_id):
        """获取目标的所有特征"""
        return self.features.get(target_id, [])

    def find_best_match(self, candidate_feature, threshold=0.7):
        """在所有保存的特征中寻找最佳匹配"""
        best_match_id = None
        max_similarity = 0.0

        for target_id, features in self.features.items():
            for feature in features:
                try:
                    similarity = np.dot(feature.flatten(), candidate_feature.flatten()) / (
                            np.linalg.norm(feature) * np.linalg.norm(candidate_feature) + 1e-10)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_id = target_id
                except:
                    continue

        if max_similarity > threshold:
            return best_match_id, max_similarity
        return None, max_similarity


# 初始化特征存储器
feature_storage = FeatureStorage()

# 姿态特征提取与相似度计算
def extract_pose_features(keypoints):
    """提取详细的姿态特征用于匹配"""
    if keypoints is None or len(keypoints) < 17:
        return None
    
    kpts = keypoints.copy()
    valid_mask = (kpts[:, 0] > 0) & (kpts[:, 1] > 0)
    valid_kpts = kpts[valid_mask]
    
    if len(valid_kpts) < 5:
        return None
    
    left_shoulder, right_shoulder = 5, 6
    left_hip, right_hip = 11, 12
    
    center_points = []
    if valid_mask[left_shoulder] and valid_mask[right_shoulder]:
        center_points.extend([kpts[left_shoulder], kpts[right_shoulder]])
    if valid_mask[left_hip] and valid_mask[right_hip]:
        center_points.extend([kpts[left_hip], kpts[right_hip]])
    
    center = np.mean(center_points, axis=0) if len(center_points) > 0 else np.mean(valid_kpts, axis=0)
    
    distances = np.linalg.norm(valid_kpts - center, axis=1)
    scale = np.max(distances) if len(distances) > 0 and np.max(distances) > 0 else 1.0
    
    features = []
    for i in range(17):
        if valid_mask[i]:
            features.extend(((kpts[i] - center) / scale).tolist())
        else:
            features.extend([0.0, 0.0])
            
    return np.array(features)

def pose_similarity(feat1, feat2):
    """计算姿态特征相似度"""
    if feat1 is None or feat2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    cos_sim = np.dot(feat1, feat2) / (norm1 * norm2)
    return max(0.0, min(1.0, cos_sim))

def draw_pose_keypoints(img, keypoints, color=(0, 255, 255)):
    """绘制姿态关键点和骨架连接"""
    if keypoints is None or len(keypoints) < 17:
        return img
    
    # COCO格式17个关键点的连接关系
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # 绘制关键点
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(img, (int(x), int(y)), 3, color, -1)
    
    # 绘制骨架连接
    for connection in skeleton:
        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # 转换为0-based索引
        if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and 
            keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and
            keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0):
            pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
            pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
            cv2.line(img, pt1, pt2, color, 2)
    
    return img


# 初始化OAK相机管道（优化版本）
def create_pipeline():
    fps = 45
    pipeline = dai.Pipeline()
    
    # 定义源和输出
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    depthOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    depthOut.setStreamName("depth")

    # 属性配置
    rgbCamSocket = dai.CameraBoardSocket.CAM_A

    # 配置RGB相机
    camRgb.setBoardSocket(rgbCamSocket)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    camRgb.setPreviewSize(640, 480)  # 输出640x480
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(fps)

    # 配置单目相机
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setFps(fps)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setFps(fps)

    # 配置立体深度
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(rgbCamSocket)
    stereo.setOutputSize(640, 480)

    # 链接
    camRgb.preview.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(depthOut.input)

    return pipeline


# 相机连接管理器
class CameraManager:
    def __init__(self, max_retries=5, retry_delay=3):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.device = None
        self.pipeline = None

    def connect_camera(self):
        """尝试连接相机，失败时重试"""
        for attempt in range(self.max_retries):
            try:
                print(f"尝试连接OAK相机... (第 {attempt + 1}/{self.max_retries} 次)")
                
                # 创建管道
                self.pipeline = create_pipeline()
                
                # 尝试连接设备
                self.device = dai.Device(self.pipeline)
                print("OAK相机连接成功！")
                return True
                
            except Exception as e:
                print(f"相机连接失败 (第 {attempt + 1} 次): {e}")
                
                # 清理资源
                if self.device:
                    try:
                        self.device.close()
                    except:
                        pass
                    self.device = None
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries - 1:
                    print(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    print("所有重试尝试都失败了")
                    return False
        
        return False

    def get_device(self):
        """获取设备实例"""
        return self.device

    def close(self):
        """关闭设备连接"""
        if self.device:
            try:
                self.device.close()
            except:
                pass
            self.device = None


# 目标状态类（优化版本）
class TargetState:
    def __init__(self, target_id):
        self.target_id = target_id
        self.position = None
        self.velocity = (0, 0)
        self.size = None
        self.confidence = 0.0
        self.timestamp = time.time()
        self.last_seen = time.time()
        self.trajectory = deque(maxlen=30)  # 减少轨迹长度
        self.world_position = None
        self.distance = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.reid_features = deque(maxlen=15)  # 减少特征保留数量
        self.stable_features = deque(maxlen=5)  # 减少稳定特征集合
        self.initial_feature = None
        self.lock_strength = 1.0
        self.stability = 0.0
        self.consecutive_frames = 0
        self.last_update_time = time.time()
        self.last_output_time = time.time()
        self.lost_frame_count = 0
        self.kalman = self.init_kalman_filter()
        self.active = True
        self.color = (0, 0, 255)  # 固定为红色
        self.full_body_feature = None
        self.last_feature_save_time = 0
        # 添加姿态特征相关属性
        self.pose_features = deque(maxlen=10)  # 姿态特征历史
        self.stable_pose_features = deque(maxlen=3)  # 稳定姿态特征
        self.initial_pose_feature = None  # 初始姿态特征

    def init_kalman_filter(self):
        kalman = cv2.KalmanFilter(4, 2)  # 简化卡尔曼滤波器
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        return kalman

    def update_reid_features(self, frame, box, is_full_body=False):
        x1, y1, x2, y2 = map(int, box)
        if y1 >= y2 or x1 >= x2 or y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            return

        # 减少扩展区域
        expand = 5
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(frame.shape[1], x2 + expand)
        y2 = min(frame.shape[0], y2 + expand)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if x1 >= x2 or y1 >= y2:
            return

        target_roi = frame[y1:y2, x1:x2]
        if target_roi.size == 0:
            return

        try:
            feature = reid_extractor.extract_features(target_roi)

            # 保存初始特征
            if self.initial_feature is None:
                self.initial_feature = feature
                self.stable_features.append(feature)
                # 保存为全身特征
                if is_full_body:
                    self.full_body_feature = feature
                    feature_storage.add_feature(self.target_id, feature)

            # 只在目标稳定时更新特征
            if self.stability > 0.6:
                self.reid_features.append(feature)

                # 当稳定性高时更新稳定特征
                if self.stability > 0.8 and len(self.stable_features) < 3:
                    self.stable_features.append(feature)

                    # 当目标稳定时保存全身特征
                    if is_full_body and self.full_body_feature is None:
                        self.full_body_feature = feature
                        feature_storage.add_feature(self.target_id, feature)
        except Exception as e:
            print(f"ReID特征更新失败: {e}")

    def compare_signature(self, frame, box):
        if not self.reid_features and not self.stable_features and self.full_body_feature is None:
            return 0.0

        x1, y1, x2, y2 = map(int, box)
        if y1 >= y2 or x1 >= x2 or y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            return 0.0

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        candidate_roi = frame[y1:y2, x1:x2]
        if candidate_roi.size == 0:
            return 0.0

        try:
            candidate_feature = reid_extractor.extract_features(candidate_roi)
            max_similarity = 0.0

            # 优先比较全身特征
            if self.full_body_feature is not None:
                similarity = np.dot(self.full_body_feature.flatten(), candidate_feature.flatten()) / (
                        np.linalg.norm(self.full_body_feature) * np.linalg.norm(candidate_feature) + 1e-10)
                if similarity > max_similarity:
                    max_similarity = similarity

            # 与初始特征比较
            if self.initial_feature is not None:
                similarity = np.dot(self.initial_feature.flatten(), candidate_feature.flatten()) / (
                        np.linalg.norm(self.initial_feature) * np.linalg.norm(candidate_feature) + 1e-10)
                if similarity > max_similarity:
                    max_similarity = similarity

            # 与稳定特征比较
            for feature in self.stable_features:
                similarity = np.dot(feature.flatten(), candidate_feature.flatten()) / (
                        np.linalg.norm(feature) * np.linalg.norm(candidate_feature) + 1e-10)
                if similarity > max_similarity:
                    max_similarity = similarity

            # 与最近特征比较
            for feature in self.reid_features:
                similarity = np.dot(feature.flatten(), candidate_feature.flatten()) / (
                        np.linalg.norm(feature) * np.linalg.norm(candidate_feature) + 1e-10)
                if similarity > max_similarity:
                    max_similarity = similarity
        except:
            max_similarity = 0.0

        return max(0.0, min(1.0, max_similarity))

    def update(self, x, y, w, h, conf, depth_map=None):
        current_time = time.time()
        dt = max(0.01, current_time - self.last_update_time)
        self.last_update_time = current_time

        self.consecutive_frames += 1
        self.stability = min(1.0, self.consecutive_frames / 20.0)  # 减少稳定帧数要求
        self.lock_strength = min(1.0, self.lock_strength + 0.1 * (0.5 + self.stability * 0.5))

        if self.position:
            prev_x, prev_y = self.position
            vx = (x - prev_x) / dt
            vy = (y - prev_y) / dt
            smooth_factor = 0.7
            self.velocity = (
                smooth_factor * vx + (1 - smooth_factor) * self.velocity[0],
                smooth_factor * vy + (1 - smooth_factor) * self.velocity[1]
            )

        self.position = (x, y)
        self.size = (w, h)
        self.confidence = conf
        self.timestamp = current_time
        self.last_seen = current_time
        self.lost_frame_count = 0
        self.trajectory.append((x, y, self.timestamp))

        if not np.isnan(x) and not np.isnan(y):
            measurement = np.array([[x], [y]], dtype=np.float32)
            self.kalman.correct(measurement)

        if depth_map is not None:
            self.world_position = calculate_3d_coordinates(depth_map, (x, y), (w, h))
            if self.world_position != (0, 0, 0) and not any(math.isnan(val) for val in self.world_position):
                X, Y, Z = self.world_position
                if Z > 0.001:
                    self.distance = math.sqrt(X**2 + Y**2 + Z**2)
                    self.yaw = math.atan2(X, Z)
                    distance_xy = math.sqrt(X**2 + Z**2)
                    self.pitch = math.atan2(Y, distance_xy) if distance_xy > 0 else 0

    def predict_next_position(self):
        prediction = self.kalman.predict()
        return prediction[0][0], prediction[1][0]

    def mark_lost(self):
        self.lost_frame_count += 1
        self.lock_strength = max(0.3, self.lock_strength - 0.05)
        self.consecutive_frames = 0
        self.stability = 0.0

        if self.lost_frame_count > 30:  # 减少丢失帧数阈值
            self.active = False

    def get_state(self):
        return {
            'id': self.target_id,
            'distance': self.distance,
            'yaw': self.yaw,
            'pitch': self.pitch,
            'position': self.position,
            'active': self.active,
            'lock_strength': self.lock_strength
        }

    def output_state(self):
        current_time = time.time()
        if current_time - self.last_output_time > 0.3 and self.active:  # 减少输出频率
            status = "活动" if self.active else f"丢失({self.lost_frame_count}帧)"
            if self.world_position and not any(math.isnan(val) for val in self.world_position):
                X, Y, Z = self.world_position
                print(f"目标ID {self.target_id} [{status}]: "
                      f"位置=({X:.2f}m, {Y:.2f}m, {Z:.2f}m), "
                      f"距离={self.distance:.2f}m, "
                      f"方位角={math.degrees(self.yaw):.2f}°, "
                      f"俯仰角={math.degrees(self.pitch):.2f}°, "
                      f"锁定强度={self.lock_strength:.2f}")
            else:
                print(f"目标ID {self.target_id} [{status}]: "
                      f"位置=({self.position[0]:.1f}, {self.position[1]:.1f}), "
                      f"锁定强度={self.lock_strength:.2f}")
            self.last_output_time = current_time

    def update_pose_features(self, keypoints):
        """更新姿态特征"""
        pose_feat = extract_pose_features(keypoints)
        if pose_feat is not None:
            self.pose_features.append(pose_feat)
            if self.initial_pose_feature is None:
                self.initial_pose_feature = pose_feat
            if self.stability > 0.8 and len(self.stable_pose_features) < 3:
                self.stable_pose_features.append(pose_feat)

    def compare_pose(self, keypoints):
        """比较姿态相似度"""
        if not any([self.initial_pose_feature is not None, self.stable_pose_features]):
             return 0.0
        
        candidate_feature = extract_pose_features(keypoints)
        if candidate_feature is None:
            return 0.0

        max_similarity = 0.0
        
        # 与初始姿态比较
        if self.initial_pose_feature is not None:
            max_similarity = max(max_similarity, pose_similarity(self.initial_pose_feature, candidate_feature))
            
        # 与稳定姿态比较
        for feature in self.stable_pose_features:
            max_similarity = max(max_similarity, pose_similarity(feature, candidate_feature))
            
        return max_similarity


# 三维坐标计算函数（优化版本）
def calculate_3d_coordinates(depth_map, center_point, size=None):
    u, v = int(center_point[0]), int(center_point[1])
    height, width = depth_map.shape

    if size is None:
        w, h = 1, 1
    else:
        w, h = size

    roi_size = max(5, min(30, w // 2, h // 2))  # 减少ROI大小
    x1 = max(0, u - roi_size)
    y1 = max(0, v - roi_size)
    x2 = min(width - 1, u + roi_size)
    y2 = min(height - 1, v + roi_size)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if x1 >= x2 or y1 >= y2:
        return (0, 0, 0)

    depth_roi = depth_map[y1:y2, x1:x2]
    valid_mask = (depth_roi > 300) & (depth_roi < 8000)

    if not np.any(valid_mask):
        return (0, 0, 0)

    valid_depths = depth_roi[valid_mask]
    median_depth = np.median(valid_depths)
    
    # 使用中值而不是加权平均以加速计算
    Z_cam = median_depth / 1000.0

    if Z_cam <= 0.3 or Z_cam > 15.0:
        return (0, 0, 0)

    # 简化相机内参
    fx = 860.0
    fy = 860.0
    cx = width / 2
    cy = height / 2

    try:
        # 相机坐标系坐标
        X_cam = (u - cx) * Z_cam / fx
        Y_cam = (v - cy) * Z_cam / fy
        
        # 转换为世界坐标系
        X_world = Z_cam
        Y_world = -X_cam
        Z_world = -Y_cam
        
    except ZeroDivisionError:
        return (0, 0, 0)

    if any(math.isnan(val) for val in (X_world, Y_world, Z_world)):
        return (0, 0, 0)

    return (X_world, Y_world, Z_world)


# 简化绘制函数（提高性能）
def draw_simple_box(img, box, target_id, conf, color, status, keypoints=None):
    x1, y1, x2, y2 = map(int, box)
    thickness = 2
    
    # 绘制边界框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # 绘制标签
    label = f"ID:{target_id} {conf:.2f}"
    if status == 'FOLLOW':
        label = f"{label} [FOLLOW]"
    
    # 绘制标签背景
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 绘制中心点
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (center_x, center_y), 3, color, -1)
    
    # 绘制姿态关键点（如果有的话）
    if keypoints is not None and status == 'FOLLOW':
        img = draw_pose_keypoints(img, keypoints, color)
    
    return img


# 多目标管理器（只跟踪ID1）
class MultiTargetManager:
    def __init__(self):
        self.targets = {}
        self.next_id = 0
        self.target_class = "person"
        self.last_detection_time = time.time()
        self.special_target_id = None  # 特殊跟踪目标ID
        self.special_target = None  # 特殊跟踪目标对象
        self.reappear_threshold = 0.7  # 重新出现匹配阈值
        self.initial_feature_start = None  # 初始特征收集开始时间
        self.initial_feature_duration = 10  # 初始特征收集持续时间（秒）

    def get_new_target_id(self):
        self.next_id += 1
        return self.next_id

    def create_target(self, frame, boxes, cls_ids, confs, depth_map, keypoints_data=None):
        # 只在首次检测时，选取最靠近中心的person目标
        if self.special_target_id is not None:
            return None
        img_h, img_w = frame.shape[:2]
        center_x, center_y = img_w / 2, img_h / 2
        min_dist = float('inf')
        best_idx = -1
        for i in range(len(boxes)):
            if model.names[cls_ids[i]] != self.target_class:
                continue
            x1, y1, x2, y2 = boxes[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        if best_idx == -1:
            return None
        # 选中中心最近目标
        x1, y1, x2, y2 = boxes[best_idx]
        conf = confs[best_idx]
        cls_id = cls_ids[best_idx]
        bbox_width, bbox_height = abs(x2 - x1), abs(y2 - y1)
        target_id = 1
        new_target = TargetState(target_id)
        new_target.update((x1 + x2) / 2, (y1 + y2) / 2, bbox_width, bbox_height, conf, depth_map)
        new_target.update_reid_features(frame, (x1, y1, x2, y2), is_full_body=True)
        
        # 更新姿态特征
        if keypoints_data is not None and best_idx < len(keypoints_data):
            keypoints = keypoints_data[best_idx]  # shape: (17, 2)
            new_target.update_pose_features(keypoints)
        
        self.initial_feature_start = time.time()
        print(f"启动10秒初始特征收集...")
        self.targets[target_id] = new_target
        self.special_target_id = target_id
        self.special_target = new_target
        print(f"创建特殊目标 ID: {target_id} 位置: ({(x1 + x2) / 2:.1f}, {(y1 + y2) / 2:.1f})")
        return new_target

    def update_target(self, target, x, y, w, h, conf, depth_map):
        target.update(x, y, w, h, conf, depth_map)

    def find_best_match(self, frame, boxes, cls_ids, confs, depth_map, keypoints_data=None):
        best_match = None
        max_match_score = 0.0
        matched_target_id = None

        # 只尝试匹配特殊目标ID1
        if self.special_target_id is None:
            return best_match, matched_target_id, max_match_score

        # 尝试匹配现有的特殊目标
        if self.special_target_id in self.targets:
            target = self.targets[self.special_target_id]
            for i in range(len(boxes)):
                cls_id = cls_ids[i]
                if model.names[cls_id] != self.target_class:
                    continue

                x1, y1, x2, y2 = boxes[i]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_width, bbox_height = abs(x2 - x1), abs(y2 - y1)

                match_score = 0.0
                
                # ReID相似度 (权重0.7)
                reid_sim = target.compare_signature(frame, (x1, y1, x2, y2))
                match_score += 0.7 * reid_sim
                
                # 姿态相似度 (权重0.2)
                pose_sim = 0.0
                if keypoints_data is not None and i < len(keypoints_data):
                    keypoints = keypoints_data[i]
                    pose_sim = target.compare_pose(keypoints)
                match_score += 0.3 * pose_sim

                # 位置相似度 (权重0.1)
                if target.position:
                    px, py = target.position
                    distance = math.sqrt((center_x - px)**2 + (center_y - py)**2)
                    position_score = max(0, 1 - distance / 300)  # 增加距离容忍度
                    match_score += 0 * position_score

                if match_score > max_match_score and match_score > 0.65:  # 稍微降低匹配阈值
                    max_match_score = match_score
                    best_match = (x1, y1, x2, y2, center_x, center_y, bbox_width, bbox_height, confs[i], cls_id, i)  # 添加索引
                    matched_target_id = self.special_target_id

        # 如果特殊目标丢失，尝试在所有检测目标中寻找重新出现的匹配
        if self.special_target_id not in self.targets or not self.targets[self.special_target_id].active:
            for i in range(len(boxes)):
                cls_id = cls_ids[i]
                if model.names[cls_id] != self.target_class:
                    continue

                x1, y1, x2, y2 = boxes[i]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_width, bbox_height = abs(x2 - x1), abs(y2 - y1)

                # 提取候选目标的特征
                candidate_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                if candidate_roi.size == 0:
                    continue

                try:
                    candidate_feature = reid_extractor.extract_features(candidate_roi)

                    # 在所有保存的特征中寻找最佳匹配
                    best_match_id, similarity = feature_storage.find_best_match(candidate_feature,
                                                                                self.reappear_threshold)

                    if best_match_id == self.special_target_id and similarity > max_match_score:
                        max_match_score = similarity
                        best_match = (x1, y1, x2, y2, center_x, center_y, bbox_width, bbox_height, confs[i], cls_id, i)
                        matched_target_id = self.special_target_id
                        print(f"重新锁定目标 ID: {self.special_target_id}, 相似度: {similarity:.2f}")
                except Exception as e:
                    print(f"重新识别特征提取失败: {e}")

        return best_match, matched_target_id, max_match_score

    def update_inactive_targets(self):
        current_time = time.time()
        for target_id, target in list(self.targets.items()):
            if not target.active:
                target.mark_lost()
                if target.lock_strength < 0.3 or current_time - target.last_seen > 8.0:  # 减少非活动目标保留时间
                    print(f"移除非活动目标 ID: {target_id}")
                    del self.targets[target_id]
            elif target.active and target.lost_frame_count > 0:
                pred_x, pred_y = target.predict_next_position()
                target.position = (pred_x, pred_y)
                target.last_seen = current_time

    def get_special_target(self):
        return self.special_target

    def output_all_states(self):
        for target in self.targets.values():
            target.output_state()

    def has_active_targets(self):
        return any(target.active for target in self.targets.values())


# 全局模型变量
model = None

# 帧捕获线程
class FrameCaptureThread(threading.Thread):
    def __init__(self, device, frame_queue):
        super().__init__()
        self.device = device
        self.frame_queue = frame_queue
        self.running = True
        self.q_rgb = None
        self.q_depth = None

    def initialize_queues(self):
        """初始化设备队列"""
        try:
            self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=True)
            self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=True)
            return True
        except Exception as e:
            print(f"队列初始化失败: {e}")
            return False

    def run(self):
        if not self.initialize_queues():
            print("帧捕获线程启动失败：无法初始化队列")
            return

        while self.running:
            try:
                in_rgb = self.q_rgb.get()
                in_depth = self.q_depth.get()
                if in_rgb is not None and in_depth is not None:
                    frame = in_rgb.getCvFrame()
                    depth_frame = in_depth.getFrame()
                    
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait() # 丢弃旧帧
                    self.frame_queue.put((frame, depth_frame))
            except Exception as e:
                print(f"相机线程错误: {e}")
                self.running = False
        print("相机线程已停止。")

    def stop(self):
        self.running = False

# 处理线程（优化版本）
class ProcessingThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, stop_event):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.target_manager = MultiTargetManager()
        self.special_target_missing_start = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()

    def run(self):
        global model
        try:
            # 使用姿态检测模型
            model = YOLO('yolo11n-pose.pt')
            print("YOLO姿态检测模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.stop_event.set()
            return

        while not self.stop_event.is_set():
            try:
                frame, depth_frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # 调整深度帧大小以匹配RGB帧 (640x480)
            if depth_frame.shape != (480, 640):
                depth_frame = cv2.resize(depth_frame, (640, 480), interpolation=cv2.INTER_NEAREST)
            depth_frame = cv2.medianBlur(depth_frame.astype(np.float32), 3).astype(np.uint16)  # 减少模糊核大小

            # --- FPS 计算 ---
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:  # 每秒计算一次FPS
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.last_fps_time = current_time
                self.frame_count = 0
                if not ENABLE_VISUALIZATION:
                    print(f"FPS: {self.fps:.1f}")

            # --- 目标检测（优化参数）---
            try:
                # 优化检测参数以提高速度
                results = model.track(
                    frame,
                    imgsz=416,  # 减小输入尺寸提高速度
                    tracker='botsort.yaml',
                    conf=0.6,  # 提高置信度阈值减少误检
                    iou=0.6,   # 提高IOU阈值减少重叠框
                    persist=True,
                    half=True,
                    verbose=False,
                    max_det=5,  # 限制最大检测数量
                    agnostic_nms=True  # 启用类别无关的NMS
                )
            except Exception as e:
                print(f"目标检测失败: {e}")
                results = []

            # 根据配置决定是否处理可视化
            if ENABLE_VISUALIZATION:
                vis_frame = frame.copy()
            
            # --- 结果处理和跟踪 ---
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                
                # 提取姿态关键点（如果有的话）
                keypoints_data = None
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    keypoints_data = results[0].keypoints.xy.cpu().numpy()  # shape: (N, 17, 2)

                # 只在首次检测时选取中心目标
                if self.target_manager.special_target_id is None:
                    new_target = self.target_manager.create_target(frame, boxes_xyxy, cls_ids, confs, depth_frame, keypoints_data)
                    if new_target and ENABLE_VISUALIZATION:
                        idx = np.argmin([((x1 + x2)/2 - frame.shape[1]/2)**2 + ((y1 + y2)/2 - frame.shape[0]/2)**2 for (x1, y1, x2, y2) in boxes_xyxy])
                        x1, y1, x2, y2 = boxes_xyxy[idx]
                        keypoints_for_draw = keypoints_data[idx] if keypoints_data is not None and idx < len(keypoints_data) else None
                        vis_frame = draw_simple_box(vis_frame, (x1, y1, x2, y2), new_target.target_id, confs[idx], new_target.color, 'NEW', keypoints_for_draw)
                else:
                    best_match, matched_target_id, match_score = self.target_manager.find_best_match(
                        frame, boxes_xyxy, cls_ids, confs, depth_frame, keypoints_data
                    )
                    if best_match and matched_target_id in self.target_manager.targets:
                        x1, y1, x2, y2, center_x, center_y, w, h, conf, cls_id, box_idx = best_match
                        target = self.target_manager.targets[matched_target_id]
                        self.target_manager.update_target(target, center_x, center_y, w, h, conf, depth_map=depth_frame)
                        
                        # 更新姿态特征
                        if keypoints_data is not None and box_idx < len(keypoints_data):
                            keypoints = keypoints_data[box_idx]
                            target.update_pose_features(keypoints)
                        
                        current_time_proc = time.time()
                        if self.target_manager.initial_feature_start is not None and current_time_proc - self.target_manager.initial_feature_start <= self.target_manager.initial_feature_duration:
                            if current_time_proc - target.last_feature_save_time >= 0.5:
                                target.update_reid_features(frame, (x1, y1, x2, y2), is_full_body=True)
                                target.last_feature_save_time = current_time_proc
                                remaining_time = self.target_manager.initial_feature_duration - (current_time_proc - self.target_manager.initial_feature_start)
                                print(f"收集初始特征 (剩余时间: {remaining_time:.1f}秒)")
                        if ENABLE_VISUALIZATION:
                            keypoints_for_draw = keypoints_data[box_idx] if keypoints_data is not None and box_idx < len(keypoints_data) else None
                            vis_frame = draw_simple_box(vis_frame, (x1, y1, x2, y2), matched_target_id, conf, target.color, 'FOLLOW', keypoints_for_draw)
                # 仅可视化其他检测目标（不跟踪）
                if ENABLE_VISUALIZATION:
                    for i in range(len(boxes)):
                        cls_id = cls_ids[i]
                        if model.names[cls_id] != self.target_manager.target_class:
                            continue
                        x1, y1, x2, y2 = boxes_xyxy[i]
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        if center_x < 50 or center_x > frame.shape[1] - 50 or center_y < 50 or center_y > frame.shape[0] - 50:
                            continue
                        # 绘制其他检测到的目标（绿色框）
                        x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        label = f"{model.names[cls_id]} {confs[i]:.2f}"
                        cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self.target_manager.update_inactive_targets()
            self.target_manager.output_all_states()

            # 绘制信息并发送结果
            if ENABLE_VISUALIZATION:
                self.draw_info(vis_frame, depth_frame)
                if self.result_queue.full():
                    self.result_queue.get_nowait()
                self.result_queue.put(vis_frame)
        
        print("处理线程已停止。")

    def draw_info(self, vis_frame, depth_frame):
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示特殊目标信息
        special_target = self.target_manager.get_special_target()
        if special_target:
            status = "活动" if special_target.active else f"丢失({special_target.lost_frame_count}帧)"
            cv2.putText(vis_frame, f"ID:{special_target.target_id} ", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示特征状态
            reid_count = len(special_target.reid_features)
            pose_count = len(special_target.pose_features)
            cv2.putText(vis_frame, f"ReID:{reid_count} Pose:{pose_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        current_time = time.time()
        if self.target_manager.initial_feature_start is not None:
            elapsed = current_time - self.target_manager.initial_feature_start
            if elapsed <= self.target_manager.initial_feature_duration:
                remaining = self.target_manager.initial_feature_duration - elapsed
                # cv2.putText(vis_frame, f"特征收集: {remaining:.1f}s", (vis_frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 简化的控制说明
        cv2.putText(vis_frame, "Q=exit | R=reset", (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def reset_tracker(self):
        self.target_manager = MultiTargetManager()
        self.special_target_missing_start = None
        print("手动重置目标选择")

# 主程序
def main():
    print(f"启动模式: {'可视化界面' if ENABLE_VISUALIZATION else '无界面'}模式")
    
    # 创建相机管理器
    camera_manager = CameraManager(max_retries=5, retry_delay=3)
    
    # 尝试连接相机
    if not camera_manager.connect_camera():
        print("无法连接到OAK相机，程序退出")
        return False
    
    device = camera_manager.get_device()
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    
    if ENABLE_VISUALIZATION:
        result_queue = queue.Queue(maxsize=1)
    else:
        result_queue = None
    
    capture_thread = None
    processing_thread = None
    
    try:
        print("OAK相机已连接，开始处理视频流...")
        if ENABLE_VISUALIZATION:
            print("可视化模式：按 'q' 退出，'r' 重置跟踪器")
        else:
            print("无界面模式：按 Ctrl+C 停止程序")
        
        capture_thread = FrameCaptureThread(device, frame_queue)
        processing_thread = ProcessingThread(frame_queue, result_queue, stop_event)

        capture_thread.start()
        processing_thread.start()

        if ENABLE_VISUALIZATION:
            # 创建显示窗口
            cv2.namedWindow('目标跟踪系统 (只跟踪ID1)', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('目标跟踪系统 (只跟踪ID1)', 1280, 720)

            # 预热阶段（减少帧数）
            print("相机预热中...")
            for i in range(5):
                try:
                    frame, _ = frame_queue.get(timeout=2)
                    # text = f"预热中... {i+1}/15"
                    # cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("目标跟踪系统 (只跟踪ID1)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_event.set()
                        break
                except queue.Empty:
                    print("预热期间未能从相机获取帧。")
                    break
            
            if not stop_event.is_set():
                print("预热完成，开始主循环。")

            # 主显示循环
            while not stop_event.is_set():
                try:
                    display_frame = result_queue.get(timeout=1)
                    cv2.imshow('目标跟踪系统 (只跟踪ID1)', display_frame)
                except queue.Empty:
                    # 如果处理线程卡住，检查它是否还活着
                    if not processing_thread.is_alive():
                        print("处理线程已意外终止。")
                        break
                    continue

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    stop_event.set()
                elif key & 0xFF == ord('r'):
                    processing_thread.reset_tracker()
        else:
            # 简单的主循环，等待中断信号
            try:
                while not stop_event.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n接收到中断信号，准备退出...")
                stop_event.set()

    except Exception as e:
        print(f"主程序运行出错: {e}")
        stop_event.set()
    
    finally:
        print("正在停止线程...")
        stop_event.set()
        
        if capture_thread:
            capture_thread.stop()
            capture_thread.join(timeout=2)
        
        if processing_thread:
            processing_thread.join(timeout=5)

        # 关闭相机连接
        camera_manager.close()
        
        if ENABLE_VISUALIZATION:
            cv2.destroyAllWindows()
        
        print("\n===== 最终跟踪报告 =====")
        if processing_thread and processing_thread.target_manager:
            tm = processing_thread.target_manager
            print(f"总跟踪目标数: {len(tm.targets)}")
            for target_id, target in tm.targets.items():
                status = "活动" if target.active else f"丢失({target.lost_frame_count}帧)"
                if target.position:
                    print(f"目标 ID {target_id}: {status}, 最后位置: ({target.position[0]:.1f}, {target.position[1]:.1f}), "
                          f"锁定强度: {target.lock_strength:.2f}")
                else:
                    print(f"目标 ID {target_id}: {status}, 无位置信息")
            # 保存所有特征
            feature_storage.save_features()
        else:
            print("未能生成报告，处理线程未正常初始化。")
    
    return True

if __name__ == "__main__":
    # 添加重试机制到主程序
    max_program_retries = 3
    for attempt in range(max_program_retries):
        print(f"\n========== 程序启动尝试 {attempt + 1}/{max_program_retries} ==========")
        
        try:
            if main():
                print("程序正常退出")
                break
            else:
                print(f"程序启动失败 (第 {attempt + 1} 次)")
        except Exception as e:
            print(f"程序运行异常 (第 {attempt + 1} 次): {e}")
        
        if attempt < max_program_retries - 1:
            print(f"等待5秒后重试...")
            time.sleep(5)
        else:
            print("所有启动尝试都失败了")
    
    sys.exit(0)